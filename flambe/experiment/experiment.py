import logging
from copy import deepcopy
from typing import Optional, Dict, Sequence, Union, List, NamedTuple, Tuple

import ray

from flambe.compile import Schema
from flambe.search import Algorithm, Search, Trial
from flambe.runner.runnable import Runnable, Environment


logger = logging.getLogger(__name__)


class Reduction(NamedTuple):
    """A reduction of the variants for source stage to k variants"""

    source: str
    k: int


class Pipeline(Schema):
    # move this to its own file if it gets big

    def __init__(self,
                 schemas: Dict[str, Schema],
                 checkpoints: Dict[str, Checkpoint]):
        """[summary]

        Parameters
        ----------
        schemas : Dict[str, Schema]
            [description]
        """
        self.schemas = schemas
        self.task = list(schemas.keys())[-1]

    def sub_pipeline(self, stage_name):
        """Return subset of the pipeline stages ending in stage_name

        The subset of pipeline stages will include all dependencies
        needed for stage the given stage and all their dependencies
        and so on.
        """
        # TODO actually prune not-needed deps
        sub_stages = {}
        for k, v in self.tasks:
            sub_stages[k] = v
            if k == stage_name:
                break
        return Pipeline(sub_stages)

    def dependencies(self):
        return self.keys()[:-1]


class Stage(object):

    def __init__(self,
                 name: str,
                 pipeline: Pipeline,
                 algorithm: Algorithm,
                 cpus_per_trial: int,
                 gpus_per_trial: int,
                 dependencies: List[List[Trial]],
                 reductions: List[Tuple[str, int]]):
        """[summary]

        Parameters
        ----------
        name : str
            [description]
        pipeline : Pipeline
            [description]
        algorithm : Algorithm
            [description]
        cpus_per_trial : int
            [description]
        gpus_per_trial : int
            [description]
        dependencies : List[List[Trial]]
            [description]
        reductions : List[Tuple[str, int]]
            [description]
        """
        self.name = name
        self.pipeline = pipeline
        self.algorithm = algorithm
        self.cpus_per_trial = cpus_per_trial
        self.gpus_per_trial = gpus_per_trial
        self.dependencies = dependencies
        self.reductions = reductions

    def run(self):
        """Execute the stage."""
        # Each dependency is the output of stage, which is
        # a pipeline object with all the variants that ran
        merge = []
        for pipeline in self.dependencies:
            # Filter errors out
            trials = filter(lambda t: not t.is_error(), pipeline.trials)
            # Find all reductions with this stage name in the pipeline
            filters = [r for r in self.reductions if r.source == result.stage_name]
            if filters:
                min_reduction = min(r.k for r in filters)
                trials = sorted(trials, key=lambda t: t.best_metric(), reverse=True)[:k]
                merge.append(result.topk(min_reduction))
            else:
                merge.append(result)

        # Take an intersection with the other sub-pipelines
        pipeline = self.pipeline.merge_union(merge)

        # Run remaining searches in parallel
        object_ids = []
        for variant in pipeline.iter_variants():
            # Set up the search
            search = ray.remote(Search).remote(
                variant,
                self.algorithm,
                self.cpus_per_trial,
                self.gpus_per_trial
            )
            # Exectue remotely
            variant_env = env.clone()
            variant_env.output_path = os.path.join(output_path, self.name)
            object_id = search.run.remote(variant_env)
            object_ids.append(object_id)

        results = ray.get(object_ids)
        pipelines = []
        for variants in results:
            # Each variant object is a dictionary from variant name
            # to a dictionary with schema, params, and checkpoint
            for var_dict in variants:
                var_pipeline = Pipeline.from_pipeline(**var_dict)
                pipelines.append(var_pipeline)

        # Merge all together into parallel options
        pipeline = Pipeline.from_union(pipelines)
        return pipeline


class Experiment(Runnable):

    def __init__(self,
                 name: str,
                 pipeline: Optional[Dict[str, Schema]] = None,
                 resources: Optional[Dict[str, str]] = None,
                 devices: Dict[str, int] = None,
                 save_path: str = 'flambe_output',
                 output_path: str = 'flambe_output',
                 algorithm: Optional[Dict[str, Algorithm]] = None,
                 reduce: Optional[Dict[str, int]] = None) -> None:
        """Iniatilize an experiment.

        Parameters
        ----------
        name : str
            [description]
        pipeline : Optional[Dict[str, Schema]], optional
            [description], by default None
        resources : Optional[Dict[str, str]], optional
            [description], by default None
        resume : Optional[Union[str, Sequence[str]]], optional
            [description], by default None
        debug : bool, optional
            [description], by default False
        devices : Dict[str, int], optional
            [description], by default None
        save_path : Optional[str], optional
            [description], by default None
        output_path : Optional[str], optional
            [description], by default None
        algorithm : Optional[Dict[str, Algorithm]], optional
            [description], by default None
        reduce : Optional[Dict[str, int]], optional
            [description], by default None
        """
        self.name = name
        self.output_path = output_path or save_path

        self.pipeline = pipeline if pipeline is not None else dict()
        self.algorithm = algorithm if algorithm is not None else dict()
        self.reduce = reduce if reduce is not None else dict()
        self.devices: Dict[str, Dict[str, int]] = dict()

    def run(self, env: Optional[Environment] = None):
        """Execute the Experiment.

        Parameters
        ----------
        env : Environment, optional
            An optional environment object.

        """
        logger.info('Experiment started.')

        env = env if env is not None else Environment(self.output_path)
        if not ray.is_initialized():
            ray.init(local_mode=env.debug)

        stage_to_id: Dict[str, int] = {}
        pipeline = Pipeline(self.pipeline)

        for name, schema in pipeline.items():
            # Get dependencies
            sub_pipeline = pipeline.sub_pipeline(name)
            depedency_ids = [stage_to_id[d] for d in sub_pipeline.dependencies()]

            # Construct the state
            stage = ray.remote(Stage).remote(
                name=name,
                pipeline=sub_pipeline,
                reductions=self.reduce[name],
                dependencies=depedency_ids,
                devices=self.devices[name],
                resume=False  # TODO
            )

            # Execute the stage remotely
            object_id = stage.run.remote()
            stage_to_id[name] = object_id

        # Wait until the extperiment is done
        ray.wait(stage_to_id.values(), num_returns=1)
        logger.info('Experiment ended.')
