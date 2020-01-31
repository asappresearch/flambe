import logging
import copy
from typing import Optional, Dict, List, NamedTuple, Tuple, Set

import ray

from flambe.compile import Schema, Link
from flambe.search import Algorithm, Search, Trial, Choice
from flambe.search.search import Checkpoint
from flambe.runner.runnable import Runnable, Environment


logger = logging.getLogger(__name__)


class Pipeline(Schema):

    def __init__(self,
                 task: Schema,
                 dependencies: Dict[str, Schema],
                 variant_ids: Optional[Dict[str, str]] = None,
                 checkpoints: Optional[Dict[str, Checkpoint]] = None):
        """[summary]

        Parameters
        ----------
        schemas : Dict[str, Schema]
            [description]
        """
        self.schemas = schemas
        self.deps = dependencies
        self.var_ids = variant_ids if variant_ids is not None else dict() 
        self.checkpoints = checkpoints if checkpoints is not None else dict()

    @property
    def task(self) -> str:
        """Get the dependencies for this pipeline.

        Returns
        -------
        List[str]
            [description]

        """
        return list(self.schemas.keys())[-1]

    @property
    def dependencies(self) -> List[str]:
        """Get the dependencies for this pipeline.

        Returns
        -------
        List[str]
            [description]

        """
        return list(self.schemas.keys())[:-1]

    def sub_pipeline(self, stage_name: str) -> 'Pipeline':
        """Return subset of the pipeline stages ending in stage_name

        The subset of pipeline stages will include all dependencies
        needed for stage the given stage and all their dependencies
        and so on.

        """

        def get_deps(schema: Schema, visited: Optional[Set[str]] = None) -> Set[str]:
            """[summary]

            Parameters
            ----------
            schema : Schema
                [description]
            visited : Optional[Set[str]], optional
                [description], by default None

            Returns
            -------
            Set[str]
                [description]

            """
            links = set()
            visited = visited if visited is not None else set()
            for path, item in Schema.traverse(schema, yield_schema='never'):
                if isinstance(item, Link):
                    stage = item.schematic_path[0]
                    links.add(stage)
                    if stage not in visited:
                        visited.add(stage)
                        new_links = get_deps(self.schemas[stage], visited)
                        links.update(new_links)
            return links

        deps = get_deps(self)
        sub_stages = {k: v for k, v in self.schemas.items() if k in deps}
        checkpoints = {k: v for k, v in self.checkpoints.items() if k in deps}
        sub_stages[self.task] = self.schemas[self.task]
        return Pipeline(sub_stages, checkpoints)

    def matches(self, other: 'Pipeline') -> bool:
        """Check for matches."""
        for key in self.schemas.keys():
            if key in other.schemas and self.var_ids[key] != other.var_ids[key]:
                return False
        return True

    def merge(self, other: 'Pipeline') -> 'Pipeline':
        """Check for matches."""
        pipeline = copy.deepcopy(self)
        pipeline.schemas.update(other.schemas)
        pipeline.var_ids.update(other.var_ids)
        pipeline.checkpoints.update(other.checkpoints)
        return pipeline
