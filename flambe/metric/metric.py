from abc import abstractmethod

import torch

from flambe.compile import Component


class Metric(Component):
    """Base Metric interface.

    Objects implementing this interface should take in a sequence of
    examples and provide as output a processd list of the same size.

    """
    @abstractmethod
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the metric over the given prediction and target.

        Parameters
        ----------
        pred: torch.Tensor
            The model predictions

        target: torch.Tensor
            The ground truth targets

        Returns
        -------
        torch.Tensor
            The computed metric

        """
        pass

    def __call__(self, *args, **kwargs):
        """Makes Featurizer a callable."""
        return self.compute(*args, **kwargs)

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        return self.__class__.__name__
