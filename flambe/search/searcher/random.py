import numpy as np
from typing import List, Dict, Any, Optional

from flambe.search.searcher.searcher import Searcher


class RandomSearcher(Searcher):
    """
    Random search samples parameterized distributions for
    hyperparameters.  At the moment, this searcher
    only supports an independent distribution for each hyperparameter.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the random searcher.

        Parameters
        ----------
        seed: Optional[int]
            Seed for the random generator.
        """

        if seed:
            np.random.seed(seed)

        super().__init__()

    def _propose_new_params(self, **kwargs) -> Dict[str, Any]:
        """
        Samples a single hyperparameter configuration
        from parameterized distributions.

        Returns
        ----------
        Dict[str, Any]
            Configuration proposed by algorithm.
        """
        return self.space.sample()

    def register_results(self, results: Dict[int, float], **kwargs):
        """
        Records performance of hyperparameter configurations, storing
        these as a dataset for internal model building.

        Parameters
        ----------
        results: Dict[int, float]
            The dictionary of results mapping parameter id to values.
        """
        pass
