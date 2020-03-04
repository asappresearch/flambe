from typing import Dict
from abc import abstractmethod

import numpy as np
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
        """Aggregates by simply storing preds and targets

        Parameters
        ----------
        state: dict
            the metric state
        args: the pred, target tuple

        Returns
        -------
        dict
            the state dict
        """
        pred, target = args
        if not state:
            state['pred'] = []
            state['target'] = []
        state['pred'].append(pred.cpu().detach())
        state['target'].append(target.cpu().detach())
        return state

    def finalize(self, state: Dict) -> float:
        """Finalizes the metric computation

        Parameters
        ----------
        state: dict
            the metric state

        Returns
        -------
        float
            The final score.
        """
        if not state:
            # call on empty state
            return np.NaN
        pred = torch.cat(state['pred'], dim=0)
        target = torch.cat(state['target'], dim=0)
        state['accumulated_score'] = self.compute(pred, target).item()
        return state['accumulated_score']

    def __call__(self, *args, **kwargs):
        """Makes Featurizer a callable."""
        return self.compute(*args, **kwargs)

    def __str__(self) -> str:
        """Return the name of the Metric (for use in logging)."""
        return self.__class__.__name__


class AverageableMetric(Metric):
    """Metric interface for averageable metrics

    Some metrics, such as accuracy, are averaged as a final step.
    This allows for a more efficient metrics computation.
    These metrics should inherit from this class.

    """

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
        dict
            The updated state (even though the update happens in-place)
        """
        score = self.compute(*args, **kwargs)
        score_np = score.cpu().detach().numpy() \
            if isinstance(score, torch.Tensor) \
            else score
        try:
            num_samples = args[0].size(0)
        except (ValueError, AttributeError):
            raise ValueError(f'Cannot get size from {type(args[0])}')
        if not state:
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
        Any
            The final score. Can be anything, depending on metric.
        """
        return state.get('accumulated_score')
