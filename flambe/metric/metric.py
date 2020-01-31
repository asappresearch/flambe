from typing import Dict
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

    def aggregate(self, state: dict, *args, **kwargs) -> Dict:
        """

        Parameters
        ----------
        state: dict
            the state dictionary
        args:
            normally pred, target
        kwargs

        Returns
        -------
        The updated state (even though the update happens in-place)
        """
        score = self.compute(*args, **kwargs)
        if isinstance(score, torch.Tensor):
            score_np = score.cpu().detach().numpy()
        else:
            score_np = score
        try:
            num_samples = args[0].size(0)
        except (ValueError, AttributeError):
            raise ValueError(f'Cannot get size from {type(args[0])}')
        if state == {}:
            state['accumulated_score'] = 0.
            state['sample_count'] = 0
        state['accumulated_score'] = \
            (state['sample_count'] * state['accumulated_score'] +
             num_samples * score_np.item()) / \
            (state['sample_count'] + num_samples)
        state['sample_count'] = state['sample_count'] + num_samples
        return state

    def finalize(self, state) -> float:
        """
        FInalizes the metric computation

        Parameters
        ----------
        state: dict
            the metric state

        Returns
        -------
        The final score
        """
        return state['accumulated_score'] if 'accumulated_score' in state else None

    def __call__(self, *args, **kwargs):
        """Makes Featurizer a callable."""
        return self.compute(*args, **kwargs)

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        return self.__class__.__name__
