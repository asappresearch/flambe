from typing import Optional, Dict
import os

import ray

import flambe
from flambe.runner.protocol import Runnable
from flambe.compile import Schema, Registrable, YAMLLoadType
from flambe.search import Algorithm
from flambe.pipeline.schema import MultiSchema
from flambe.pipeline.stage import Stage


@ray.remote
def get_stage(name,
              pipeline,
              algorithm,
              reductions,
              cpus_per_trial,
              gpus_per_trial,
              environment,
              *dependencies):
    """Helper method launch the stage and set the envrionment."""
    stage = Stage(
        name=name,
        pipeline=pipeline,
        algorithm=algorithm,
        reductions=reductions,
        dependencies=dependencies,  # Passing object ids sets the order of computation
        cpus_per_trial=cpus_per_trial,
        gpus_per_trial=gpus_per_trial
    )
    # Modify the output path
    output_path = os.path.join(environment.output_path, name)
    flambe.set_env(env=environment, output_path=output_path)
    return stage.run()


class Pipeline(Registrable):

    def __init__(self,
                 name: Optional[str] = None,
                 stages: Optional[Dict[str, Schema]] = None,
                 algorithm: Optional[Dict[str, Algorithm]] = None,
                 reduce: Optional[Dict[str, int]] = None,
                 cpus_per_trial: Dict[str, int] = None,
                 gpus_per_trial: Dict[str, int] = None) -> None:
        """Iniatilize a Pipeline.

        Parameters
        ----------
        name : str
            The name of the pipeline to run.
        stages : Optional[Dict[str, Schema]], optional
            A set of Comparable schemas, possibly including links.
        algorithm : Optional[Dict[str, Algorithm]], optional
            A set of hyperparameter search algorithms, one for each
            defined stage. Defaults to grid searching for all.
        reduce : Optional[Dict[str, int]], optional
            A set of reduce operations, one for each defined stage.
            Defaults to no reduction.
        cpus_per_trial : int
            The number of CPUs to allocate per trial.
            Note: if the object you are searching over spawns ray
            remote tasks, then you should set this to 1.
        gpus_per_trial : int
            The number of GPUs to allocate per trial.
            Note: if the object you are searching over spawns ray
            remote tasks, then you should set this to 0.

        """
        self.name = name
        self.stages = stages if stages is not None else dict()
        self.algorithm = algorithm if algorithm is not None else dict()
        self.reduce = reduce if reduce is not None else dict()
        self.cpus_per_trial: Dict[str, int] = cpus_per_trial if cpus_per_trial is not None \
            else dict()
        self.gpus_per_trial: Dict[str, int] = gpus_per_trial if gpus_per_trial is not None \
            else dict()

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        """Provide the YAML loading rule."""
        return YAMLLoadType.KWARGS

    def run(self) -> bool:
        """Execute the Pipeline.

        Returns
        -------
        bool
            True until execution is over. Always ``False``.

        """
        # Set up envrionment
        env = flambe.get_env()
        flambe.utils.ray.initialize(env)

        stages: Dict[str, int] = {}
        pipeline = MultiSchema(self.stages)
        # Construct and execute stages as a DAG
        for name, schema in pipeline.arguments.items():
            if isinstance(schema.callable_, type) and not issubclass(schema.callable_, Runnable):
                # If we know that the schema will
                # produce a task, don't run it
                continue
            # Get dependencies
            sub_pipeline = pipeline.sub_pipeline(name)
            depedency_ids = [stages[d] for d in sub_pipeline.dependencies if d in stages]

            # Construct the stage
            object_id = get_stage.remote(
                name,
                sub_pipeline,
                self.algorithm.get(name, None),
                self.reduce,
                self.cpus_per_trial.get(name, 1),
                self.gpus_per_trial.get(name, 0),
                env,
                *depedency_ids,  # Passing object ids sets the order of computation
            )

            # Keep the object id, to use as future dependency
            stages[name] = object_id

        # Wait until the experiment is done
        # TODO progress tracking
        ray.get(list(stages.values()))

        # This is a single step Task
        _continue = False
        return _continue
