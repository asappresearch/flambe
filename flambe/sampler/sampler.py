from abc import abstractmethod
from typing import Iterator, Sequence, Tuple

import torch

from flambe.compile import Component


class Sampler(Component):
    """Base Sampler interface.

    Objects implementing this interface should implement two methods:

        - *sample*: takes a set of data and returns an iterator
        - *lenght*: takes a set of data and return the length of the
                    iterator that would be given by the sample method

    Sampler objects are used inside the Trainer to provide the data to
    the models. Note that pushing the data to the appropriate device
    is usually done inside the Trainer.

    """
    @abstractmethod
    def sample(self,
               data: Sequence[Sequence[torch.Tensor]],
               n_epochs: int = 1) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Sample from the list of features and yields batches.

        Parameters
        ----------
        data: Sequence[Sequence[torch.Tensor, ...]]
            The input data to sample from
        n_epochs: int, optional
            The number of epochs to run in the output iterator.

        Yields
        ------
        Iterator[Tuple[Tensor]]
            A batch of data, as a tuple of Tensors

        """
        pass

    @abstractmethod
    def length(self, data: Sequence[Sequence[torch.Tensor]]) -> int:
        """Return the number of batches in the sampler.

        Parameters
        ----------
        data: Sequence[Sequence[torch.Tensor, ...]]
            The input data to sample from

        Returns
        -------
        int
            The number of batches that would be created per epoch

        """
        pass
