import copy
from typing import Optional, Dict, List, Callable, Set, Any, Tuple

from flambe.compile import Schema, UnpreparedLinkError
from flambe.search import Checkpoint, Choice
from flambe.runner import get_env


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
            A mapping from stage name to corresponding schema.
        variant_ids : Optional[Dict[str, str]]
            An optional mapping from stage name to variant id.
        checkpoints : Optional[Dict[str, Checkpoint]]
            An optional mapping from stage name to checkpoint.

        """
        super().__init__(callable_=pipeline_builder, kwargs=schemas, apply_defaults=False)

        # Precompute dependencies for each stage
        self.deps: Dict[str, Set] = dict()
        for stage_name in self.arguments:
            self._update_deps(stage_name)

        self.schemas = schemas
        self.var_ids = variant_ids if variant_ids is not None else dict()
        self.checkpoints = checkpoints if checkpoints is not None else dict()
        self.error = False
        self.metric = None

    def _update_deps(self, stage_name: str):
        """Compute the set of dependencies for this stage.

        Parameters
        ----------
        stage_name : str
            The stage to consider.

        """
        schema = self.arguments[stage_name]
        if isinstance(schema, Choice):
            for option in schema.options:
                self.deps.update(option.deps)
        else:
            immediate_deps = set()
            immediate_deps = set(map(lambda x: x.schematic_path[0], schema.extract_links()))
            immediate_deps -= {stage_name}
            self.deps[stage_name] = immediate_deps
            for dep_name in list(immediate_deps):
                self.deps[stage_name] |= self.deps[dep_name]

    @property
    def task(self) -> Optional[str]:
        """Get the stage that will be returned when initialized.

        Returns
        -------
        str
            The stage name to be executed.

        """
        if self.arguments:
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

    @property
    def is_subpipeline(self) -> bool:
        """Return whether this is a complete pipeline.

        Returns
        -------
        bool
            ``True`` if the pipeline is complete

        """
        if self.task is not None:
            return len(self.deps[self.task]) == (len(self.schemas) - 1)
        else:
            return False

    def initialize(self,
                   path: Optional[Tuple[str]] = None,
                   cache: Optional[Dict[str, Any]] = None,
                   root: Optional['Schema'] = None) -> Any:
        """Override initialization to load checkpoints."""
        # Check Links
        checked = []
        for stage_name, schema in self.arguments.items():
            checked.append(stage_name)
            for link in schema.extract_links():
                if link.schematic_path[0] not in checked:
                    raise UnpreparedLinkError(f"{link} in stage '{stage_name}' doesn't point "
                                              "to preceding or current stage")
        cache = {}
        for stage_name, checkpoint in self.checkpoints.items():
            val = checkpoint.get()
            cache[stage_name] = val
        resources = get_env().local_resources
        cache.update(resources)
        return super().initialize(cache=cache)

    def sub_pipeline(self, stage_name: str) -> 'Pipeline':
        """Return subset of the pipeline stages ending in stage_name.

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
        for key in self.var_ids.keys():
            if key in other.var_ids and self.var_ids[key] != other.var_ids[key]:
                return False
        return True

    def set_param(self, path: Optional[Tuple[str]], value: Any):
        """Set path in schema to value

        Convenience method for setting a value deep in a schema. For
        example `root.set_param(('a', 'b', 'c'), val)` is the
        equivalent of `root['a']['b']['c'] = val`. NOTE: you can only
        use set_param on existing paths in the schema. If `c` does not
        already exist in the above example, a `KeyError` is raised.

        Parameters
        ----------
        path : Optional[Tuple[str]]
            Description of parameter `path`.
        value : Any
            Description of parameter `value`.

        Raises
        -------
        KeyError
            If any value in the path does not exist as the name of a
            child schema

        """
        if path is not None and tuple(path) == ('__dependencies',):
            # Trick to get rid of reset attributes
            schemas = dict(value.schemas)
            schemas[self.task] = self.schemas[self.task]  # type: ignore
            Pipeline.__init__(self, value.schemas, value.var_ids, value.checkpoints)
        else:
            super().set_param(path, value)  # type: ignore

    def __deepcopy__(self, memo=None) -> 'Pipeline':
        """Override deepcopy."""
        pipeline = Pipeline(
            schemas=copy.deepcopy(self.schemas),
            variant_ids=copy.deepcopy(self.var_ids),
            checkpoints=copy.deepcopy(self.checkpoints)
        )
        pipeline.metric = self.metric
        pipeline.error = self.error
        return pipeline

    @classmethod
    def from_yaml(cls, raw_obj: Any, callable_override: Optional[Callable] = None) -> Any:
        """Override to disable."""
        raise NotImplementedError('Pipeline YAML not supported')
