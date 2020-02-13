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
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.KWARGS_OR_POSARGS
