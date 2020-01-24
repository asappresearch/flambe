import itertools
from typing import List, Dict, Any
import numpy as np

from flambe.search.searcher.searcher import Space
from flambe.search.searcher.searcher import Searcher


class GridSearcher(Searcher):
    """
    Grid search does a brute force search over all
    discretized hyperparameter configurations.
    """

    def assign_space(self, space: Space):
        """
        Assign a space of distributions for the searcher to search
        over.  Also creates the parameters proposed by the algorithm.

        Parameters
        ----------
        space: Space
            A Space object that holds the distributions to search over.
        """
        super().assign_space(space)

        params_options = {name: d.options for name, d in self.space.distributions.items()}
        self.params_sets = self._dict_to_cartesian_list(params_options)

    @classmethod
    def _check_space(self, space: Space):
        """
        Check if the space is valid for this algorithm, which does not
        allow continuous distributions.

        Parameters
        ----------
        space: Space
            A Space object that holds the distributions to search over.
        """
        if np.any(np.array(space.var_types) == 'continuous'):
            raise ValueError('For grid search, all dimensions must be `discrete` or `choice`!')

    def _dict_to_cartesian_list(self, d: Dict[Any, List[Any]]) -> List[Dict[Any, Any]]:
        """
        Generates a list of the Cartesian product between
        hyperparameter options contained in dictionary format.

        Parameters
        ----------
        d: Dict[Any, List[Any]]
            Dictionary of choices.

        Returns
        ----------
        List[Dict[Any, Any]]
            List of hyperparameter configurations.
        """
        lst = []
        for key in d:
            lst.append([(key, val) for val in d[key]])
        cartesian_lst = list(itertools.product(*lst))[::-1]
        return cartesian_lst

    def _propose_new_params(self) -> Optional[Dict[str, Any]]:
        """
        Returns a hyperparameter configuration from the
        list of Cartesian products.

        Returns
        ----------
        Optional[Dict[str, Any]]
            New parameters proposed by algorithm.
        """
        if bool(self.params_sets):
            new_params = dict(self.params_sets.pop())
            return new_params
        else:
            return None

    def register_results(self, results: Dict[int, float], **kwargs):
        """
        Empty method because the grid search algorithm does not record
        results.

        Parameters
        ----------
        results: Dict[int, float]
            The dictionary of parameter configurations and corresponding
            results.
        """
        pass
