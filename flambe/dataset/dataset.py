from abc import abstractmethod
from typing import Sequence

from flambe import Component


class Dataset(Component):
    """Base Dataset interface.

    Dataset objects offer the main interface to loading data into the
    experiment pipepine. Dataset objects have three attributes:
    `train`, `dev`, and `test`, each pointing to a list of examples.

    Note that Datasets should also be "immutable", and as such,
    `__setitem__` and `__delitem__` will raise an error. Although this
    does not mean that the object will not be mutated in other ways,
    it should help avoid issues now and then.

    """

    @property
    @abstractmethod
    def train(self) -> Sequence[Sequence]:
        """Returns the training data as a sequence of examples."""
        pass

    @property
    @abstractmethod
    def val(self) -> Sequence[Sequence]:
        """Returns the validation data as a sequence of examples."""
        pass

    @property
    @abstractmethod
    def test(self) -> Sequence[Sequence]:
        """Returns the test data as a sequence of examples."""
        pass

    def __setitem__(self):
        """Raise an error."""
        raise ValueError("Dataset objects are immutable")

    def __delitem__(self):
        """Raise an error."""
        raise ValueError("Dataset objects are immutable")
