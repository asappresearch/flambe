from abc import abstractmethod
from typing import Any, Tuple

from flambe.compile import Options, YAMLLoadType


class Distribution(Options):

    @abstractmethod
    def sample(self) -> Any:
        """Sample from the distribution."""
        pass

    def named_sample(self) -> Tuple[str, Tuple[str, Any]]:
        """Sample from the distribution, and name the option."""
        sample = self.sample()
        return str(sample), sample

    @classmethod
    def from_sequence(cls, args) -> 'Distribution':
        """Build the distribution from positonal arguments."""
        return cls(*args)  # type: ignore

    @classmethod
    def from_dict(cls, **kwargs) -> 'Distribution':
        """Build the distribution from keyword arguments."""
        return cls(**kwargs)  # type: ignore

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.KWARGS_OR_ARG
