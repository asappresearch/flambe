from abc import abstractmethod
from typing import Any, Dict

from flambe.compile.schema import Options


class Distribution(Options):

    @abstractmethod
    def sample(self) -> Any:
        """Sample from the distribution."""
        pass

    @classmethod
    def from_sequence(cls, args) -> 'Distribution':
        """Build the distribution from positonal arguments."""
        return cls(args)  # type: ignore

    @classmethod
    def from_dict(cls, **kwargs) -> 'Distribution':
        """Build the distribution from keyword arguments."""
        return cls(**kwargs)  # type: ignore

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Convert to yaml."""
        return representer.represent_sequence(tag, node.elements)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str, tag: str) -> 'Distribution':
        """Convert from yaml."""
        args, = list(constructor.construct_yaml_seq(node))
        if factory_name is None:
            if isinstance(args, (tuple, list)):
                return cls.from_sequence(args)  # type: ignore
            elif isinstance(args, Dict):
                return cls.from_dict(**args)
        else:
            factory = getattr(cls, factory_name)
            return factory(args)
