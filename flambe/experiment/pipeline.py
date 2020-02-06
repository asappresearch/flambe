import copy
from typing import Optional, Dict, List, Callable, Set, Any

from flambe.compile import Schema, UnpreparedLinkError
from flambe.search.search import Checkpoint


def pipeline_builder(**kwargs):
    """Get the last initialized object in the pipeline."""
    return list(kwargs.values())[-1]


class Pipeline(Schema):
    """Schema specialized to represent a series of stateful stages

    NOTE: All arguments of this schema must also be schemas, which is
    not a requirement for Schema in general.

    """

    def __init__(self,
                 schemas: Dict[str, Schema],
                 variant_ids: Optional[Dict[str, str]] = None,
                 checkpoints: Optional[Dict[str, Checkpoint]] = None):
        """Initialize a Pipeline.

        Parameters
        ----------
        schemas : Dict[str, Schema]
            Description of parameter `schemas`.
        variant_ids : Optional[Dict[str, str]]
            Description of parameter `variant_ids` (default is None).
        checkpoints : Optional[Dict[str, Checkpoint]]
            Description of parameter `checkpoints` (default is None).

        """
        for stage_name, schema in schemas.items():
            if not isinstance(schema, Schema):
                raise TypeError(f'Value at {stage_name} is not a Schema')

        # TODO check keys in variants and checkpoints
        super().__init__(callable=pipeline_builder, kwargs=schemas, apply_defaults=False)
        # Check Links
        links = {}
        checked = []
        for stage_name, schema in self.arguments.items():
            checked.append(stage_name)
            for link in schema.extract_links():
                if link.schematic_path[0] not in checked:
                    raise UnpreparedLinkError(f"{link} in stage '{stage_name}' doesn't point "
                                              "to preceding or current stage")

        # Precompute dependencies for each stage
        self.deps: Dict[str, Set] = dict()
        for stage_name in iter(self):
            self._update_deps(stage_name)

        last_stage = list(schemas.keys())[-1]
        self.is_subpipeline = len(self.deps[last_stage]) == (len(schemas) - 1)
        self.var_ids = variant_ids if variant_ids is not None else dict()
        self.checkpoints = checkpoints if checkpoints is not None else dict()
        self.error = False
        self.metric = None

    def _update_deps(self, stage_name: str):
        """[summary]

        Parameters
        ----------
        stage_name : str
            [description]

        """
        schema = self.arguments[stage_name]
        immediate_deps = set(map(lambda x: x.schematic_path[0], schema.extract_links()))
        immediate_deps -= {stage_name}
        self.deps[stage_name] = immediate_deps
        for dep_name in immediate_deps:
            self.deps[stage_name] |= self.deps[dep_name]

    @property
    def task(self) -> Optional[str]:
        """Get the stage that will be returned when initialized

        Returns
        -------
        str
            The stage name to be executed.

        """
        if self.is_subpipeline:
            return list(self.arguments.keys())[-1]
        else:
            return None

    @property
    def dependencies(self) -> List[str]:
        """Get the dependencies for this pipeline.

        Returns
        -------
        List[str]
            A list of dependencies, as stage names.

        """
        if self.task is not None:
            return list(self.deps[self.task])
        else:
            return []

    def sub_pipeline(self, stage_name: str) -> 'Pipeline':
        """Return subset of the pipeline stages ending in stage_name

        The subset of pipeline stages will include all dependencies
        needed for stage the given stage and all their dependencies
        and so on.

        Parameters
        ----------
        stage_name: str
            The name of the stage to construct the subpipeline over.

        Returns
        -------
        Pipeline
            The output subpipeline.

        """
        deps = self.deps[stage_name]
        sub_stages = {k: v for k, v in self.arguments.items() if k in deps}
        var_ids = {k: v for k, v in self.var_ids.items() if k in deps}
        checkpoints = {k: v for k, v in self.checkpoints.items() if k in deps}
        sub_stages[stage_name] = self.arguments[stage_name]
        subpipeline = Pipeline(sub_stages, var_ids, checkpoints)
        assert subpipeline.is_subpipeline
        return subpipeline

    def append(self,
               name: str,
               schema: Schema,
               var_id: Optional[str] = None,
               checkpoint: Optional[Checkpoint] = None):
        """Append a new schema.

        Parameters
        ----------
        schema : Schema
            [description]
        var_id : Optional[str], optional
            [description], by default None
        checkpoint : Optional[Checkpoint], optional
            [description], by default None

        """
        if name in self.arguments:
            raise ValueError(f"{name} already in pipeline.")
        self.arguments[name] = schema
        self._update_deps(name)
        if var_id is not None:
            self.var_ids[name] = var_id
        if checkpoint is not None:
            self.checkpoints[name] = checkpoint

    def merge(self, other: 'Pipeline') -> 'Pipeline':
        """Updates internals with the provided pipeline.

        Parameters
        ----------
        other: Pipeline
            The pipeline to merge in.

        Returns
        -------
        Pipeline
            A new merged pipeline.

        """
        stages = copy.deepcopy(self.arguments).update(copy.deepcopy(other.arguments))
        var_ids = copy.copy(self.var_ids).update(other.var_ids)
        checkpoints = copy.deepcopy(self.checkpoints).update(copy.deepcopy(other.checkpoints))
        return Pipeline(schemas=stages, variant_ids=var_ids, checkpoints=checkpoints)

    def flatten(self) -> 'Pipeline':
        """Flatten the pipeline.

        Flattening a pipeline means taking all child pipelines
        and surfacing their stages at the top level. This method
        is used to construct the pipelines post-execution.

        Returns
        -------
        Pipeline
            A flattened versiion of this pipeline.

        """
        schemas: Dict[str, Schema] = dict()
        checkpoints: Dict[str, Checkpoint] = dict()
        var_ids: Dict[str, str] = dict()

        for name, schema in self.arguments.items():
            if isinstance(schema, Pipeline):
                schemas.update(schema.arguments)
                checkpoints.update(schema.checkpoints)
                var_ids.update(schema.var_ids)
            else:
                schemas[name] = schema

        return Pipeline(schemas, var_ids, checkpoints)

    def matches(self, other: 'Pipeline') -> bool:
        """Check for matches.

        A match occurs when all matching stage names have the same
        variant id. This method is used to construct pipelines in
        the Stage object.

        Parameters
        ----------
        other : Pipeline
            The pipeline to compare with.

        Returns
        -------
        bool
            Whether the given pipeline is a match.

        """
        for key in self.arguments.keys():
            if key in other.arguments and self.var_ids[key] != other.var_ids[key]:
                return False
        return True

    @classmethod
    def from_yaml(cls,
                  constructor: Any,
                  node: Any,
                  factory_name: str,
                  tag: str,
                  callable: Callable) -> Any:
        """Override to disable."""
        raise NotImplementedError('Pipeline YAML not supported')
