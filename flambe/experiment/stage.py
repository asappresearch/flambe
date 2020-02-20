from typing import Dict, List, Optional
import logging

from flambe.search import Algorithm, Search, Searchable, Choice, Trial, Checkpoint
from flambe.experiment.pipeline import Pipeline


logger = logging.getLogger(__name__)


class Stage(object):
    """A stage in the Experiment pipeline.

    This object is a wrapper around the Search object, which adds
    logic to support hyperparameter searches as nodes in a directed
    acyclic graph. In particular, it handles applying dependency
    resolution, running searches, and reducing to the best trials.

    """

    def __init__(self,
                 name: str,
                 pipeline: Pipeline,
                 dependencies: List[Dict[str, Pipeline]],
                 cpus_per_trial: int,
                 gpus_per_trial: int,
                 algorithm: Optional[Algorithm] = None,
                 reductions: Optional[Dict[str, int]] = None):
        """Initialize a Stage.

        Parameters
        ----------
        name : str
            A name for this stage in the experiment pipeline.
        pipeline : Pipeline
            The sub-pipeline to execute in this stage.
        dependencies : List[Dict[str, Any]]
            A list of previously executed pipelines.
        cpus_per_trial : int
            The number of CPUs to allocate per trial.
            Note: if the object you are searching over spawns ray
            remote tasks, then you should set this to 1.
        gpus_per_trial : int
            The number of GPUs to allocate per trial.
            Note: if the object you are searching over spawns ray
            remote tasks, then you should set this to 0.
        algorithm : Algorithm, optional
            An optional search algorithm.
        reductions : Dict[str, int], optional
            Reductions to apply between stages.

        """
        self.name = name
        self.pipeline = pipeline
        self.algorithm = algorithm
        self.cpus_per_trial = cpus_per_trial
        self.gpus_per_trial = gpus_per_trial
        self.reductions = reductions if reductions is not None else dict()

        # Flatten out the dependencies
        self.dependencies = {name: p for dep in dependencies for name, p in dep.items()}

        # Get the non-searchable stages in the pipeline
        searchables, non_searchables = [], []
        for name, schema in list(pipeline.schemas.items())[:-1]:
            if isinstance(schema.callable_, type) and \
               issubclass(schema.callable_, Searchable):  # type: ignore
                searchables.append(name)
            else:
                non_searchables.append(name)

        searchable_deps = {dep for n in searchables for dep in pipeline.deps[n]}
        self.task_dependencies = [n for n in non_searchables if n not in searchable_deps]

    def filter_dependencies(self, pipelines: Dict[str, 'Pipeline']) -> Dict[str, 'Pipeline']:
        """Filter out erros, and apply reductions on dependencies.

        Parameters
        ----------
        pipelines: Dict[str, Pipeline]
            The dependencies, as previously executed sub-pipelines.

        Returns
        -------
        Dict[str, Pipeline]
            An updated list of pipelines.

        """
        # Filter out error trials
        pipelines = {k: p for k, p in pipelines.items() if not p.error}

        for stage, reduction in self.reductions.items():
            # Find all pipelines with the given stage name
            reduce = {k: p for k, p in pipelines.items() if p.task == stage}
            ignore = {k: p for k, p in pipelines.items() if p.task != stage}

            # Apply reductions
            keys = sorted(reduce.keys(), key=lambda k: reduce[k].metric, reverse=True)
            pipelines = {k: reduce[k] for k in keys[:reduction]}
            pipelines.update(ignore)

        return pipelines

    def merge_variants(self, pipelines: Dict[str, 'Pipeline']) -> Dict[str, 'Pipeline']:
        """Merge based on conditional dependencies.

        Parameters
        ----------
        pipelines: Dict[str, Pipeline]
            The dependencies, as previously executed sub-pipelines.

        Returns
        -------
        Dict[str, Pipeline]
            An updated list of pipelines.

        """
        variants: Dict[str, Pipeline] = dict()

        for name, pipe in pipelines.items():
            match_found = False
            for var_name, var in variants.items():
                # If all the matching stages have matching schemas
                if pipe.matches(var):
                    match_found = True
                    variants[var_name] = pipe.merge(var)

            # If no match was found, then just add to variants
            if not match_found:
                variants[name] = pipe

        return variants

    def construct_pipeline(self, pipelines: Dict[str, 'Pipeline']) -> Optional['Pipeline']:
        """Construct the final pipeline.

        May return ``None`` in cases where there were no complete
        pipelines available for this stage.

        Parameters
        ----------
        pipelines: List[Pipeline]
            The dependencies, as previously executed sub-pipelines.

        Returns
        -------
        Pipeline, optional
            The pipeline to exectute, or None.

        """
        # Filter out not complete pipelines
        pipelines = {k: v for k, v in pipelines.items() if v.is_subpipeline}

        # Get list of new stages to add
        append_stages = self.task_dependencies
        append_stages.append(self.name)
        schemas = {name: self.pipeline[name] for name in append_stages}

        # Here we nest the pipelines so that we can search over
        # cross-stage parameter configurations
        if len(pipelines) == 0:
            try:
                pipeline = Pipeline(schemas)
                out = pipeline if pipeline.is_subpipeline else None
            except Exception:
                logger.info(f"Found incomplete pipeline at stage {self.name}")
                out = None
        elif len(pipelines) == 1:
            pipeline = pipelines[list(pipelines.keys())[0]]
            schemas.update(pipeline.schemas)
            try:
                pipeline = Pipeline(schemas, pipeline.var_ids, pipeline.checkpoints)
                out = pipeline if pipeline.is_subpipeline else None
            except Exception:
                logger.info(f"Found incomplete pipeline at stage {self.name}")
                out = None
        else:
            # Search over previous variants
            schemas = dict(__dependencies=Choice(pipelines), **schemas)
            out = Pipeline(schemas)

        return out

    def run(self) -> Dict[str, Pipeline]:
        """Execute the stage.

        Proceeds as follows:

        1. Filter out errored trials and apply reductions
        2. Construct dependency variants
        3. Get the link types between this stage and its dependencies
        4. Construct the pipelines to execute
        5. For every pipeline launch a Search remotely
        6. Aggregate results, and return executed pipelines

        Returns
        -------
        Dict[str, Pipeline]
            A list of pipelines each containing the schema, checkpoint,
            variant id, and error status for the respective trial.

        """
        # Each dependency is the output of stage, which is
        # a pipeline object with all the variants that ran
        pipelines = self.filter_dependencies(self.dependencies)

        # Take an intersection with the other sub-pipelines
        pipelines = self.merge_variants(pipelines)

        # Construct pipeline to execute
        pipeline = self.construct_pipeline(pipelines)
        if pipeline is None:
            logger.warn(f"Stage {self.name} did not have any variants to execute.")
            return dict()

        # Exectue the search
        search = Search(
            pipeline,
            self.algorithm,
            self.cpus_per_trial,
            self.gpus_per_trial
        )
        trials: Dict[str, Trial] = search.run()

        # Each variants object is a dictionary from variant name
        # to a dictionary with schema, params, and checkpoint
        pipelines: Dict[str, Pipeline] = dict()
        for name, trial in trials.items():
            # Flatten out the pipeline schema
            variant: Optional[Pipeline] = trial.get_schema()  # type: ignore
            if variant is None:
                raise ValueError(f"No schema set on output pipeline {name}")

            var_id: Optional[str] = trial.get_var_id()
            if var_id is None:
                raise ValueError(f"No variant id set on output pipeline {name}")

            checkpoint: Optional[Checkpoint] = trial.get_checkpoint()
            if checkpoint is None:
                raise ValueError(f"No variant id set on output pipeline {name}")
            variant.var_ids[self.name] = var_id
            variant.checkpoints[self.name] = checkpoint
            variant.error = trial.is_error()
            variant.metric = trial.best_metric
            pipelines[name] = variant

        return pipelines
