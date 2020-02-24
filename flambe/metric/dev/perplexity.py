from typing import Dict
import numpy as np
import torch

from flambe.metric import Metric


class Perplexity(Metric):
    """Token level perplexity, computed a exp(cross_entropy)."""

    def __init__(self):
        """Perplexity, computed as CrossEntropy"""
        self.entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the preplexity given the input and target.

        Parameters
        ----------
        pred: torch.Tensor
            input logits of shape (B x N)
        target: torch.LontTensor
            target tensor of shape (B)

        Returns
        -------
        torch.float
            Output perplexity

        """
        entropy = self.entropy(pred, target).mean()
        return torch.exp(entropy)

    def aggregate(self, state: dict, *args, **kwargs) -> Dict:
        """Aggregates by only storing entropy per sample

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
            state['historic_score'] = []
        state['historic_score'].append(self.entropy(pred, target))
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
        entropy = torch.cat(state['historic_score'], dim=0)
        return torch.exp(entropy.mean()).item()
