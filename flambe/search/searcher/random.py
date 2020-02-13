import numpy as np
from typing import Dict, Any, Optional

from flambe.search.distribution import Distribution
from flambe.search.searcher.searcher import Space, Searcher


class RandomSearcher(Searcher):
    """
    Random search samples parameterized distributions for
    hyperparameters.  At the moment, this searcher
    only supports an independent distribution for each hyperparameter.
    """

    def __init__(self, space: Dict[str, Distribution], seed: Optional[int] = None):
        """Initialize the random searcher.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Dictionary mapping variable names to their initial
            distributions.
        seed: Optional[int]
            Seed for the random generator.

        """
        if seed:
            np.random.seed(seed)
        super().__init__(space)

    def check_space(self, space: Space):
        """Check if the space is valid for this algorithm.

        Everything is valid.

        Parameters
        ----------
        space: Space
            A Space object that holds the distributions to search over.

        """
        pass

    def _propose_new_params_in_model_space(self) -> Optional[Dict[str, Any]]:
        """Samples a single hyperparameter configuration.

        Returns
        ----------
        Dict[str, Any], optional
            Configuration proposed by algorithm.

        """
        return self.space.sample()
