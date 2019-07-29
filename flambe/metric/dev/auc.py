import torch
import sklearn.metrics
import numpy as np

from flambe.metric.metric import Metric


class AUC(Metric):

    def __init__(self, max_fpr=1.0):
        """Initialize the AUC metric.

        Parameters
        ----------
        max_fpr : float, optional
            Maximum false positive rate to compute the area under

        """
        self.max_fpr = max_fpr

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute AUC at the given max false positive rate.

        Parameters
        ----------
        pred : torch.Tensor
            The model predictions
        target : torch.Tensor
            The binary targets

        Returns
        -------
        torch.Tensor
            The computed AUC

        """
        scores = np.array(pred)
        targets = np.array(target)

        # Case when number of elements added are 0
        if not scores.size or not targets.size:
            return torch.tensor(0.5)

        fpr, tpr, _ = sklearn.metrics.roc_curve(targets, scores, sample_weight=None)

        # Compute the area under the curve using trapezoidal rule
        max_index = np.searchsorted(fpr, [self.max_fpr], side='right').item()
        area = np.trapz(tpr[:max_index], fpr[:max_index])

        return torch.tensor(area / self.max_fpr).float()
