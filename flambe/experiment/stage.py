import os
import copy
from typing import Dict, List

import ray

from flambe.search import Algorithm, Search, Choice
from flambe.experiment.pipeline import Pipeline
from flambe.runner import Environment


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
                 algorithm: Algorithm,
                 cpus_per_trial: int,
                 gpus_per_trial: int,
                 dependencies: List[Pipeline],
                 reductions: Dict[str, int],
                 envrionmnent: Environment):
        """Initialize a Stage.

        Parameters
        ----------
        name : str
            A name for this stage in the experiment pipeline.
        pipeline : Pipeline
            The sub-pipeline to execute in this stage.
        algorithm : Algorithm
            A search algorithm.
        cpus_per_trial : int
            The number of CPUs to allocate per trial.
            Note: if the object you are searching over spawns ray
            remote tasks, then you should set this to 1.
        gpus_per_trial : int
            The number of GPUs to allocate per trial.
            Note: if the object you are searching over spawns ray
            remote tasks, then you should set this to 0.
        dependencies : List[Pipeline]
            A list of previously executed pipelines.
        reductions : Dict[str, int]
            Reductions to apply between stages.

        """
        self.name = name
        self.pipeline = pipeline
        self.algorithm = algorithm
        self.cpus_per_trial = cpus_per_trial
        self.gpus_per_trial = gpus_per_trial
        self.dependencies = dependencies
        self.reductions = reductions
        self.env = envrionmnent

    def filter_dependencies(self, pipelines: List['Pipeline']) -> List['Pipeline']:
        """Filter out erros, and apply reductions on dependencies.

        Parameters
        ----------
        pipelines: List[Pipeline]
            The dependencies, as previously executed sub-pipelines.

        Returns
        -------
        List[Pipeline]
            An updated list of pipelines with error filtering and
            reductions applied.

        """
        # Filter out error trials
        pipelines = [p for p in pipelines if not p.error]

        for stage, reduction in self.reductions.items():
            # Find all pipelines with the given stage name
            reduce = [p for p in pipelines if p.task == stage]
            ignore = [p for p in pipelines if p.task != stage]

            # Apply reductions
            reduce = sorted(reduce, key=lambda p: p.metric, reverse=True)
            pipelines = reduce[:reduction] + ignore

        return pipelines

    def merge_variants(self, pipelines: List['Pipeline']) -> List['Pipeline']:
        """Filter out erros, and apply reductions on dependencies.

        Parameters
        ----------
        pipelines: List[Pipeline]
            The dependencies, as previously executed sub-pipelines.

        Returns
        -------
        List[Pipeline]
            An updated list of pipelines with error filtering and
            reductions applied.

        """
        variants: List[Pipeline] = []

        for pipe in pipelines:
            match_found = False
            for i, var in enumerate(variants):
                # If all the matching stages have matching schemas
                if pipe.matches(var):
                    match_found = True
                    variants[i] = pipe.merge(var)
            # If no match was found, then just add to variants
            if not match_found:
                variants.append(pipe)

        return variants

    def construct_pipelines(self, pipelines: List['Pipeline']) -> List['Pipeline']:
        """Filter out erros, and apply reductions on dependencies.

        Parameters
        ----------
        pipelines: List[Pipeline]
            The dependencies, as previously executed sub-pipelines.

        Returns
        -------
        List[Pipeline]
            An updated list of pipelines with error filtering and
            reductions applied.

        """
        schemas = copy.deepcopy(self.pipeline)
        task = schemas[self.name]

        # Get link types
        link_types: Dict[str, str] = dict()
        for link in task.extract_links():
            link_type = 'choice'  # TODO: 'choice' if isinstance(link, LinkChoice) else 'variant'
            name = link.schematic_path[0]
            if name in link_types and link_types[name] != link_type:
                raise ValueError("{self.name}: Links to the same stage must be of the same type.")
            link_types[name] = link_type

        # Constructe final pipelines
        # Here we nest the pipelines so that we can search over
        # cross-stages parameter configurations

        # TODO: actually read out the links types correctly
        pipeline = Pipeline({
            'dependencies': Choice(pipelines),  # type: ignore
            self.name: task
        })

        return [pipeline]

    def run(self) -> List[Pipeline]:
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
        List[Pipeline]
            A list of pipelines each containing the schema, checkpoint,
            variant id, and error status for the respective trial.

        """
        # Each dependency is the output of stage, which is
        # a pipeline object with all the variants that ran
        filtered = self.filter_dependencies(self.dependencies)

        # Take an intersection with the other sub-pipelines
        merged = self.merge_variants(filtered)

        # Construct pipelines to execute
        variants = self.construct_pipelines(merged)

        # Run remaining searches in parallel
        object_ids = []
        for variant in variants:
            # Set up the search
            search = ray.remote(Search).remote(
                variant,
                self.algorithm,
                self.cpus_per_trial,
                self.gpus_per_trial
            )
            # Exectue remotely
            variant_env = self.env.clone()
            variant_env.output_path = os.path.join(self.env.output_path, self.name)
            object_id = search.run.remote(variant_env)
            object_ids.append(object_id)

        # Get results and construct output pipelines
        results = ray.get(object_ids)
        pipelines = []
        for variants in results:
            # Each variants object is a dictionary from variant name
            # to a dictionary with schema, params, and checkpoint
            for var_dict in variants:
                # Flatten out the pipeline schema
                pipeline: Pipeline = var_dict['schema'].flatten()
                # Add search results to the pipeline
                pipeline.var_ids.update(var_dict['var_id'])
                pipeline.checkpoints.update(var_dict['checkpoint'])
                pipeline.error = var_dict['error']
                pipeline.metric = var_dict['metric']
                pipelines.append(pipeline)

        return pipelines
