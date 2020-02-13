from abc import abstractmethod
from typing import Any

from flambe.compile import Options, YAMLLoadType


class Distribution(Options):

    @abstractmethod
    def sample(self) -> Any:
        """Sample from the distribution."""
        pass

    def name(self, sample: Any) -> str:
        """Sample from the distribution, and name the option."""
        return str(sample)

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.KWARGS_OR_POSARGS
