from typing import Tuple, Optional, Callable, Dict, List, Any
import os
import subprocess
import copy
import getpass

import ray
import torch

from flambe.runner import Runnable, Environment
from flambe.compile.schema import Schema, Variants
from flambe.search.trial import Trial
from flambe.search.searchable import Searchable
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
        host : Optional[str], optional
            An optional host to upload the checkpoint to,
            by default None

        """
        self.path = os.path.expanduser(path)
        self.checkpoint_path = os.path.join(self.path, 'checkpoint.pt')
        self.remote = f"{user}@{host}:{self.checkpoint_path}" if host else None

    def get(self) -> Searchable:
        return
        if os.path.exists(self.checkpoint_path):
            searchable = torch.load(self.checkpoint_path)
        else:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            if self.remote:
                subprocess.run(f'rsync -a {self.remote} {self.checkpoint_path}')
                searchable = torch.load(self.checkpoint_path)
            else:
                raise ValueError(f"Checkpoint {self.checkpoint_path} couldn't be found.")
        return searchable

    def set(self, searchable):
        return
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(searchable, self.checkpoint_path)
        if self.remote:
            subprocess.run(f'rsync -a {self.checkpoint_path} {self.remote}')


class SearchableAdapter:
    """Perform computation of a task."""

    def __init__(self, schema: Schema, checkpoint: Checkpoint) -> None:
        """Initialize the Trial."""
        self.schema = schema
        self.searchable: Optional[Searchable] = None
        self.checkpoint = checkpoint

    def step(self) -> Tuple[bool, Optional[float]]:
        """Run a step of the Trial"""
        if self.searchable is None:
            self.searchable = self.schema()

        continue_ = self.searchable.step()
        metric = self.searchable.metric()
        self.checkpoint.set(self.searchable)
        return continue_, metric

    def kill(self) -> None:
        """Kill the trial."""
        ray.actor.exit_actor()


class CallableAdapter:
    """Perform computation of a function."""

    def __init__(self, schema: Schema) -> None:
        """Initialize the Trial."""
        self.schema = schema
        self.callable: Optional[Callable] = None

    def step(self) -> Tuple[bool, Optional[float]]:
        """Run a step of the Trial."""
        self.callable = self.callable or self.schema()
        continue_ = False
        metric = self.callable()
        return continue_, metric

    def kill(self) -> None:
        """Kill the trial."""
        ray.actor.exit_actor()


class Search(Runnable):
    """Implement a hyperparameter search over any schema.

    Use a Search to construct a hyperparameter search over any
    objects. Takes as input a schema to search, and algorithm,
    and resources to allocate per variant.

    Example
    -------

    >>> schema = Schema(func, arg1=uniform(0, 1), arg2=choice('a', 'b'))
    >>> search = Search(schema, algorithm=Hyperband())
    >>> search.run()

    """
    def __init__(self,
                 schema: Schema,
                 algorithm: Optional[Algorithm] = None,
                 cpus_per_trial: int = 1,
                 gpus_per_trial: int = 0,
                 output_path: str = 'flambe__output',
                 refresh_waitime: float = 1.0,
                 use_object_store: bool = False) -> None:
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
        refresh_waitime : float, optional
            The minimum amount of time to wait before a refresh.
            Defaults ``10`` seconds.
        use_object_store: bool, optional
            Whether to use the plasma object store to move objects
            between workers. See Ray documentation for more details.
            Default ``False``.

        """
        if not ray.is_initialized():
            ray.init()

        self.output_path = output_path
        self.n_cpus = cpus_per_trial
        self.n_gpus = gpus_per_trial
        self.refresh_waitime = refresh_waitime

        total_resources = ray.cluster_resources()
        if self.n_cpus > 0 and self.n_cpus > total_resources['CPU']:
            raise ValueError("# of CPUs required per trial is larger than the total available.")
        elif self.n_gpus > 0 and self.n_gpus > total_resources['GPU']:
            raise ValueError("# of CPUs required per trial is larger than the total available.")

        # Check schema
        if any(isinstance(dist, Variants) for dist in schema.extract_search_space().values()):
            raise ValueError("Schema cannot contain grid options, please split first.")

        self.schema = schema
        self.algorithm = algorithm if algorithm is not None else GridSearch()

    def run(self, env: Optional[Environment] = None) -> Dict[str, Dict]:
        """Execute the search.

        Parameters
        ----------
        env : Environment, optional
            An optional environment object.

        Returns
        -------
        Dict[str, Dict]
            A list of trial objects.

        """
        env = env if env is not None else Environment(self.output_path)

        if isinstance(self.schema.callable, type) and issubclass(self.schema.callable, Searchable):
            checkpointable = True
            adapter = ray.remote(num_cpus=self.n_cpus, num_gpus=self.n_gpus)(SearchableAdapter)
        else:
            checkpointable = False
            adapter = ray.remote(num_cpus=self.n_cpus, num_gpus=self.n_gpus)(CallableAdapter)

        search_space = self.schema.extract_search_space().items()
        self.algorithm.initialize(dict((".".join(k), v) for k, v in search_space))  # type: ignore

        running: List[int] = []
        finished: List[int] = []

        trials: Dict[str, Trial] = dict()
        state: Dict[str, Dict[str, Any]] = dict()
        object_id_to_trial_id: Dict[str, str] = dict()

        while running or (not self.algorithm.is_done()):
            # Get all the current object ids running
            if running:
                finished, running = ray.wait(running, timeout=self.refresh_waitime)

            # Process finished trials
            for object_id in finished:
                _continue, metric = ray.get(object_id)
                trial_id = object_id_to_trial_id[str(object_id)]

                # Checkpoint
                if checkpointable:
                    actor = state[trial_id]['actor']
                    checkpoint = state[trial_id]['checkpoint']
                    checkpoint.set(actor)

                # Check if terminated
                trial = trials[trial_id]
                if _continue:
                    trial.set_metric(metric)
                    trial.set_has_result()
                else:
                    trial.set_terminated()

            # Compute maximum number of trials to create
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
                elif trial.is_terminated() and trial_id in state:
                    state[trial_id]['actor'].kill()
                    del state[trial_id]['actor']
                elif trial.is_created():
                    schema_copy = copy.deepcopy(self.schema)
                    space = dict((tuple(k.split('.')), v) for k, v in trial.parameters.items())
                    schema_copy.set_from_search_space(space)
                    state[trial_id] = dict()
                    if checkpointable:
                        checkpoint = Checkpoint(
                            path=os.path.join(env.ouput_path, trial.generate_name()),
                            host=env.head_node_ip,
                            user=getpass.getuser()
                        )
                        state[trial_id]['checkpoint'] = checkpoint
                        state[trial_id]['actor'] = adapter.remote(schema_copy, checkpoint)
                    else:
                        state[trial_id]['actor'] = adapter.remote(schema_copy)
                    trial.set_resume()

                # Launch created and resumed
                if trial.is_resuming():
                    object_id = state[trial_id]['actor'].step.remote()
                    object_id_to_trial_id[str(object_id)] = trial_id
                    running.append(object_id)
                    trial.set_running()

        result = dict()
        for trial_id, trial in trials.items():
            checkpoint = state[trial_id].get('checkpoint', None)
            result[trial.generate_name()] = dict(trial=trial, checkpoint=checkpoint)

        return result
