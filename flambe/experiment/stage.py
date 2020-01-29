import copy
from typing import Optional, Dict, List, NamedTuple, Tuple, Set

import ray

from flambe.search import Algorithm, Search, Trial, Choice
from flambe.search.search import Checkpoint
from flambe.experiment.pipeline import Pipeline


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

    def filter_dependencies(self, pipelines: List['Pipeline']) -> List['Pipeline']:
        """Merge pipelines the given pipelines into self.

        For each
        """
        out = []

        for pipeline in pipelines:
            # Filter errors out
            trials = filter(lambda t: not t.is_error(), pipeline.trials)
            # Find all reductions with this stage name in the pipeline
            filters = [r for r in self.reductions if r.source == result.stage_name]
            if filters:
                min_reduction = min(r.k for r in filters)
                trials = sorted(trials, key=lambda t: t.best_metric(), reverse=True)[:k]
                out.append(result.topk(min_reduction))
            else:
                out.append(result)

        return out

    def construct_variants(self, pipelines: List['Pipeline']) -> List['Pipeline']:
        """Merge pipelines the given pipelines into self.

        For each
        """
        schemas = copy.deepcopy(self.pipeline)
        stages = list(schemas.keys())[::-1]

        # Get link types
        link_types: Dict[str, str] = dict()
        for link in schemas[self.name].extract_links():
            link_type = 'choice' if isinstance(link, LinkChoice) else 'variant'
            name = link.schematic_path[0]
            if name in link_types and link_types[name] != link_type:
                raise ValueError("{self.name}: Links to the same stage must be of the same type.")
            link_types[name] = link_type

        # Perform merge algorithm
        variants: List[Pipeline] = []
        for pipe in pipelines:
            found_match = False
            for var in variants:
                if pipe.matches(var):
                    found_match = True
                    variants.append(pipe.merge(var))
            if not found_match:
                variants.append(pipe)

        # Add the new stage to the above variants
        return variants

    def run(self):
        """Execute the stage."""
        # Each dependency is the output of stage, which is
        # a pipeline object with all the variants that ran
        filtered = self.filter(self.dependencies)
        # Take an intersection with the other sub-pipelines
        variants = self.construct_variants(filtered)

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
            variant_env = env.clone()
            variant_env.output_path = os.path.join(output_path, self.name)
            object_id = search.run.remote(variant_env)
            object_ids.append(object_id)

        # Get results and construct output pipelines
        results = ray.get(object_ids)
        pipelines = []
        for variants in results:
            # Each variant object is a dictionary from variant name
            # to a dictionary with schema, params, and checkpoint
            for var_dict in variants:
                # Flatten out the pipeline schema
                schema: Pipeline = var_dict['schema'].flatten()
                schema.checkpoints.update(var_dict['checkpoint'])
                schema.var_ids.update(var_dict.update('var_ids'))
                pipelines.append(schema)

        return pipelines
