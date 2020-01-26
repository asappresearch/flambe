import itertools
from typing import List, Dict, Any, Optional

from flambe.search.distribution import Distribution, Continuous
from flambe.search.searcher.searcher import Space
from flambe.search.searcher.searcher import Searcher


class GridSearcher(Searcher):
    """A grid searcher.

    Grid search does a brute force search over all
    discretized hyperparameter configurations.

    """

    def __init__(self, space: Dict[str, Distribution]):
        """Assign a space of distributions to search over.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Dictionary mapping variable names to their initial
            distributions.

        """
        super().__init__(space)

        dists = self.space.dists
        params_options: Dict[str, List] = {n: d.options for n, d in dists.items()}  # type: ignore
        self.params_sets = self._dict_to_cartesian_list(params_options)

    def check_space(self, space: Space):
        """Check if the space is valid for this algorithm.

        Does not allow continuous distributions.

        Parameters
        ----------
        space: Space
            A Space object that holds the distributions to search over.

        """
        if any(isinstance(d, Continuous) for d in space.dists.values()):
            raise ValueError('For grid search, all dimensions must be `discrete` or `choice`!')

    def _dict_to_cartesian_list(self, d: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generates a list of the Cartesian product of the options.

        Parameters
        ----------
        d: Dict[Any, List[Any]]
            Dictionary of choices.

        Returns
        ----------
        List[Dict[str, Any]]
            List of hyperparameter configurations.

        """
        lst = []
        for key in d:
            lst.append([(key, val) for val in d[key]])
        cartesian_lst = list(itertools.product(*lst))[::-1]
        return cartesian_lst

    def _propose_new_params_in_model_space(self) -> Optional[Dict[str, Any]]:
        """Propose a new hyperparameter configuration.

        Returns a hyperparameter configuration from the
        list of Cartesian products.

        Returns
        -------
        Dict[str, Any], optional
            The configuration proposed by the searcher.

        """
        if bool(self.params_sets):
            new_params = dict(self.params_sets.pop())
            return new_params
        else:
            return None
