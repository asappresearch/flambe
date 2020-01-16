import logging
from copy import deepcopy
from typing import Optional, Dict, Sequence, Union, List, NamedTuple, Tuple

import ray

from flambe.compile import Schema
from flambe.search import Algorithm, Search, Trial
from flambe.runner.runnable import Environment


logger = logging.getLogger(__name__)


class Reduction(NamedTuple):
    """A reduction of the variants for source stage to k variants"""

    source: str
    k: int


class Pipeline(Schema):
    # move this to its own file if it gets big

    def __init__(self, tasks: Dict[str, Schema]):
        self.tasks = tasks
        self.task = list(tasks.keys())[-1]

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

    def step(self):
        return self.task.step()

    def save_state(self):
        self.task.save_state()

    def load_state(self):
        self.task.load_state()


class Stage(object):

    def __init__(self,
                 name: str,
                 full_pipeline_id: int,
                 algorithm: Algorithm,
                 cpus_per_trial: int,
                 gpus_per_trial: int,
                 dependencies: List[List[Trial]],
                 reductions: List[Tuple[str, int]]):
        self.name = name
        self.full_pipeline_id = full_pipeline_id
        self.algorithm = algorithm
        self.cpus_per_trial = cpus_per_trial
        self.gpus_per_trial = gpus_per_trial
        self.dependencies = dependencies
        self.reductions = reductions

    def run(self):
        # Get full pipeline
        pipeline = ray.get(self.full_pipeline_id)
        # Fetch dependencies
        results = ray.get(self.dependencies)

        # Mask out failed trials and apply reductions
        successful = []
        for trials in results:
            # Filter errors out
            trials = filter(lambda t: not t.is_error(), trials)
            # Select topk
            filtered = [r for r in self.reductions if r.source == result.stage_name]
            if filtered:
                min_reduction = min(r.k for r in filtered)
                trials = sorted(trials, key=lambda x: trials.best_metric(), reverse=True)[:k]
                successful.append(result.topk(min_reduction))
            else:
                successful.append(result)

        # Take an intersection with the other sub-pipelines
        pipeline.merge_intersect(*successful)

        # Run remaining searches in parallel
        result_ids = []
        for variant in pipeline.iter_variants():
            # Construct and execute the search
            search = Search.remote(variant,
                                   self.algorithm,
                                   self.cpus_per_trial,
                                   self.gpus_per_trial)
            sampled_variants_id = search.run.remote()
            # Remote run will return id mapping to list of variants
            result_ids.append(sampled_variants_id)
        # Each id maps to list of variant schemas
        list_of_variants = ray.get(result_ids)
        # Flatten list of lists of variants
        all_variant_schemas = [item.schema for sublist in list_of_variants for item in sublist]
        # Merge all together into parallel options
        pipeline.merge_union(*all_variant_schemas)
        # Put final merged pipeline in ray and return result id
        final_result_id = ray.put(pipeline)
        return final_result_id


class Experiment(object):

    def __init__(self,
                 name: str,
                 pipeline: Optional[Dict[str, Schema]] = None,
                 resources: Optional[Dict[str, str]] = None,
                 resume: Optional[Union[str, Sequence[str]]] = None,
                 debug: bool = False,
                 devices: Dict[str, int] = None,
                 save_path: Optional[str] = None,
                 algorithm: Optional[Dict[str, Algorithm]] = None,
                 reduce: Optional[Dict[str, int]] = None,
                 env: Environment = None) -> None:
        self.name = name
        self.save_path = save_path

        self.pipeline = Pipeline(pipeline or dict())
        self.algorithm = algorithm or dict()
        self.reduce = reduce or dict()
        self.devices = dict()

    def add_stage(self,
                  name: str,
                  schema: Schema,
                  algorithm: Optional[Algorithm] = None,
                  reduce: Optional[int] = None,
                  n_cpus_per_trial: int = 1,
                  n_gpus_per_trial: int = 0) -> None:
        self.pipeline[name] = schema
        self.algorithm[name] = deepcopy(algorithm)
        if reduce is not None:
            self.reduce[name] = reduce
        self.devices[name] = {'cpu': n_cpus_per_trial, 'gpu': n_gpus_per_trial}

    def run(self, resume: bool = False) -> None:
        logger.info('Experiment started.')
        stage_to_result: Dict[str, int] = {}
        full_pipeline_id = ray.put(self.pipeline)

        for name in self.pipeline:
            # Get dependencies as a list of result object ids
            pipeline = self.pipeline.sub_pipeline(name)
            depedency_ids = [stage_to_result[d] for d in pipeline.dependencies()]

            stage = ray.remote(Stage).remote(name,
                                             full_pipeline_id,
                                             self.reduce[name],
                                             self.devices[name],
                                             dependencies=depedency_ids,
                                             resume=resume)
            result = stage.run.remote()
            stage_to_result[name] = result

        # Wait until the extperiment is done
        ray.wait(stage_to_result.values(), num_returns=1)
        logger.info('Experiment ended.')
