from typing import Tuple, Optional, Dict, List, Any
import os
import subprocess
import copy
import getpass
import warnings
import socket

import ray
import torch

from flambe.runner import Environment
from flambe.compile import Registrable, YAMLLoadType, Schema, GridVariants
from flambe.search.trial import Trial
from flambe.search.protocol import Searchable
from flambe.search.algorithm import Algorithm, GridSearch


class Checkpoint(object):

    def __init__(self,
                 path: str,
                 host: Optional[str] = None,
                 user: Optional[str] = None):
        """Initialize a checkpoint.

        Parameters
        ----------
        path : str
            The local path used for saving
        host : str, optional
            An optional host to upload the checkpoint to,
            by default None
        user: str, optional
            An optional user to use alongside the host name,
            by default None

        """
        self.path = path
        self.host = host
        self.checkpoint_path = os.path.join(self.path, 'checkpoint.pt')
        self.remote = f"{user}@{host}:{self.checkpoint_path}" if host else None

    def get(self) -> Searchable:
        """Retrieve the object from a checkpoint.

        Returns
        -------
        Searchable
            The restored Searchable object.

        """
        if os.path.exists(self.checkpoint_path):
            searchable = torch.load(self.checkpoint_path)
        else:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            if self.remote:
                subprocess.run(f'rsync -az -e "ssh -i $HOME/ray_bootstrap_key.pem" \
                    {self.remote} {self.checkpoint_path}')
                searchable = torch.load(self.checkpoint_path)
            else:
                raise ValueError(f"Checkpoint {self.checkpoint_path} couldn't be found.")
        return searchable

    def set(self, searchable: Searchable):
        """Retrieve the object from a checkpoint.

        Parameters
        ----------
        Searchable
            The Searchable object to save.

        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(searchable, self.checkpoint_path)
        if self.remote:
            current_ip = socket.gethostbyname(socket.gethostname())
            if str(current_ip) != self.host:
                subprocess.run(f'rsync -az -e "ssh -i $HOME/ray_bootstrap_key.pem" \
                    {self.checkpoint_path} {self.remote}')


@ray.remote
class RayAdapter:
    """Perform computation of a task."""

    def __init__(self, schema: Schema, checkpoint: Checkpoint) -> None:
        """Initialize the Trial."""
        self.schema = schema
        self.searchable: Optional[Searchable] = None
        self.checkpoint = checkpoint

    # @ray.remote(num_return_values=2)
    def step(self) -> Any:
        """Run a step of the Trial"""
        if self.searchable is None:
            self.searchable = self.schema()

        continue_ = self.searchable.step()
        metric = self.searchable.metric()
        self.checkpoint.set(self.searchable)
        return continue_, metric


class Search(Registrable):
    """Implement a hyperparameter search over any schema.

    Use a Search to construct a hyperparameter search over any
    objects. Takes as input a schema to search, and algorithm,
    and resources to allocate per variant.

    Example
    -------

    >>> schema = Schema(cls, arg1=uniform(0, 1), arg2=choice('a', 'b'))
    >>> search = Search(schema, algorithm=Hyperband())
    >>> search.run()

    """

    def __init__(self,
                 schema: Schema,
                 algorithm: Optional[Algorithm] = None,
                 cpus_per_trial: int = 1,
                 gpus_per_trial: int = 0,
                 output_path: str = 'flambe_output',
                 refresh_waitime: float = 1.0) -> None:
        """Initialize a hyperparameter search.

        Parameters
        ----------
        schema : Schema[Task]
            A schema of the callable or Task to search
        algorithm : Algorithm
            The hyperparameter search algorithm
        cpus_per_trial : int
            The number of cpu's to allocate per trial
        gpus_per_trial : int
            The number of gpu's to allocate per trial
        output_path: str, optional
            An output path for this search
        refresh_waitime : float, optional
            The minimum amount of time to wait before a refresh.
            Defaults ``1`` seconds.

        """
        self.output_path = output_path
        self.n_cpus = cpus_per_trial
        self.n_gpus = gpus_per_trial
        self.refresh_waitime = refresh_waitime

        # Check schema
        if any(isinstance(dist, GridVariants) for dist in schema.extract_search_space().values()):
            raise ValueError("Schema cannot contain grid options, please split first.")

        self.schema = schema
        self.algorithm = GridSearch() if algorithm is None else algorithm

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.KWARGS

    def run(self, env: Optional[Environment] = None) -> List[Dict[str, Any]]:
        """Execute the search.

        Parameters
        ----------
        env : Environment, optional
            An optional environment object.

        Returns
        -------
        List[Dict[str, Any]
            A dictionary pointing from trial name to sub-dicionary
            containing the keys 'trial' and 'checkpoint' pointing to
            a Trial and a Checkpoint object respecitvely.

        """
        env = env if env is not None else Environment(self.output_path)
        if not ray.is_initialized():
            ray.init(address="auto", local_mode=env.debug)

        if not env.debug:
            total_resources = ray.cluster_resources()
            if self.n_cpus > 0 and self.n_cpus > total_resources['CPU']:
                raise ValueError("# of CPUs required per trial is larger than the total available.")
            elif self.n_gpus > 0 and self.n_gpus > total_resources['GPU']:
                raise ValueError("# of CPUs required per trial is larger than the total available.")

        search_space = self.schema.extract_search_space().items()
        self.algorithm.initialize(dict((".".join(k), v) for k, v in search_space))  # type: ignore
        running: List[int] = []
        finished: List[int] = []

        trials: Dict[str, Trial] = dict()
        state: Dict[str, Dict[str, Any]] = dict()
        object_id_to_trial_id: Dict[str, str] = dict()
        trial_id_to_object_id: Dict[str, str] = dict()

        while running or (not self.algorithm.is_done()):
            # Get all the current object ids running
            if running:
                finished, running = ray.wait(running, timeout=self.refresh_waitime)

            # Process finished trials
            for object_id in finished:
                trial_id = object_id_to_trial_id[str(object_id)]
                trial = trials[trial_id]
                try:
                    _continue, metric = ray.get(object_id)
                    if _continue:
                        trial.set_metric(metric)
                        trial.set_has_result()
                    else:
                        trial.set_terminated()
                except Exception as e:
                    trial.set_error()
                    warnings.warn(f"Trial {trial_id} failed.")
                    if env.debug:
                        warnings.warn(str(e))

            # Compute maximum number of trials to create
            if env.debug:
                max_queries = int(all(t.is_terminated() for t in trials.values()))
            else:
                current_resources = ray.available_resources()
                max_queries = current_resources.get('CPU', 0) // self.n_cpus
                if self.n_gpus:
                    n_possible_gpu = current_resources.get('GPU', 0) // self.n_gpus
                    max_queries = min(max_queries, n_possible_gpu)

            # Update the algorithm and get new trials
            trials = self.algorithm.update(trials, maximum=max_queries)
            # Update based on trial status
            for trial_id, trial in trials.items():
                # Handle creation and termination
                if trial.is_paused() or trial.is_running():
                    continue
                elif trial.is_terminated() and 'actor' in state[trial_id]:
                    del state[trial_id]['actor']
                elif trial.is_created():
                    schema_copy = copy.deepcopy(self.schema)
                    space = dict((tuple(k.split('.')), v) for k, v in trial.parameters.items())
                    schema_copy.set_from_search_space(space)

                    # Update state
                    trial.set_resume()
                    state[trial_id] = dict()
                    state[trial_id]['schema'] = schema_copy
                    checkpoint = Checkpoint(
                        path=os.path.join(env.output_path, trial_id),
                        host=env.head_node_ip,
                        user=getpass.getuser()
                    )
                    state[trial_id]['checkpoint'] = checkpoint
                    state[trial_id]['actor'] = RayAdapter.remote(  # type: ignore
                        schema=schema_copy,
                        checkpoint=checkpoint
                    )

                # Launch created and resumed
                if trial.is_resuming():
                    object_id = state[trial_id]['actor'].step.remote()
                    object_id_to_trial_id[str(object_id)] = trial_id
                    trial_id_to_object_id[trial_id] = str(object_id)
                    running.append(object_id)
                    trial.set_running()

        # Construct result output
        results = []
        for trial_id, trial in trials.items():
            results.append({
                'schema': state[trial_id].get('schema'),
                'checkpoint': state[trial_id].get('checkpoint', None),
                'error': trial.is_error(),
                'metric': trial.best_metric,
                # The ray object id is guarenteed to be unique
                'var_id': str(trial_id_to_object_id[trial_id])
            })
        return results
