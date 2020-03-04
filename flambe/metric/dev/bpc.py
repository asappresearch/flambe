from typing import Dict

import numpy as np
import torch

from flambe.metric.dev.perplexity import Perplexity


class BPC(Perplexity):
    """Bits per character. Computed as log_2(perplexity)

    Inherits from Perplexity to share aggregate functionality.
    """

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the bits per character given the input and target.

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
        return torch.log2(torch.exp(entropy))

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
        if not state or state['sample_count'] == 0:
            # call on empty state
            return np.NaN
        return torch.log2(torch.exp(state['accumulated_score'] / state['sample_count'])).item()
