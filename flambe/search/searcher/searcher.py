from abc import ABC, abstractmethod

from typing import Dict, Any, List, Union, Tuple

import numpy as np

from flambe.search.distribution import Distribution


class Space(object):
    """Search space object.

    Manages the various dimensions that the hyperparameters are in.
    This object is used internally by the search aglrothms and should
    not be used directly.

    """

    def __init__(self, distributions: Dict[str, Distribution]):
        """
        Initialize the search space with a set of search distributions.

        Parameters
        ----------
        distributions: Dict[str, Distribution]
            Dictionary mapping variable names to their initial
            distributions.
        """
        self.distributions = distributions
        self.var_names = list(distributions.keys())
        self.dists = [distributions[name] for name in self.var_names]
        self.n_vars = len(self.dists)

        self.var_types = [dist.var_type for dist in self.dists]
        self.var_bounds = [(dist.var_min, dist.var_max) if dist.is_numerical  # type: ignore
                           else (np.nan, np.nan) for dist in self.dists]

        self.name2idx = {name: i for i, name in enumerate(self.var_names)}

    def sample(self) -> Dict[str, Any]:
        """Sample from the search space.

        Returns
        ----------
        Dict[str, Any]
            A sample from each distribution of the searcher.
        """
        samp = {}
        for name, dist in self.distributions.items():
            samp[name] = dist.sample()
        return samp

    def sample_raw_dist(self) -> Dict[str, Any]:
        """
        Randomly sample from the raw distributions
        without applying their transforms.

        Returns
        ----------
        Dict[str, Any]
            A sample from each raw distribution of the searcher.
        """
        samp = {}
        for name, dist in self.distributions.items():
            if dist.is_numerical:
                s = dist.sample_raw_dist()
            else:
                s = dist.sample()
            samp[name] = s
        return samp

    def apply_transform(self, samp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to a sample drawn from the space
        using sample_dist.

        Parameters
        ----------
        samp: Dict[str, Any]
            Dictionary of hyperparameter names and values.

        Returns
        ----------
        Dict[str, Any]
            Dictionary of hyperparameter names and transformed values.
        """
        transformed = {}
        for var_name in samp.keys():
            dist = self.dists[self.name2idx[var_name]]
            if dist.is_numerical:
                t = dist.transform_fn(samp[var_name])
            else:
                t = samp[var_name]
            transformed[var_name] = t
        return transformed

    def round_to_space(self, hp_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Round hyperparameters to possible choices for ordered variables.

        Parameters
        ----------
        hp_dict: Dict[str, Any]
            Dictionary of hyperparameter names and values.

        Returns
        ----------
        Dict[str, Any]
            Dictionary of hyperparameter names and rounded values.
        """
        rounded = {}
        for name in hp_dict.keys():
            dist = self.dists[self.name2idx[name]]
            val = hp_dict[name]

            if dist.var_type == 'discrete':
                val = dist.round_to_options(val)

            rounded[name] = val
        return rounded

    def normalize_to_space(self,
                           hp_dict: Dict[str, Any],
                           return_as_list: bool = False) ->
                           Union[Dict[Any], List[Any]]:
        """
        Normalize hyperparameters to [0, 1] based on range for
        numerical variables and convert them to integers for
        categorical variables.

        Parameters
        ----------
        hp_dict: Dict[str, Any]
            Dictionary of hyperparameter names and values.
        return_as_list: bool
            Whether or not to return the hyperparameters as a list.
            Defaults to False and returns a dictionary.

        Returns
        ----------
        Dict[str, Any]
            Dictionary of hyperparameter names and values.
        """
        normalized = {}
        for name in hp_dict.keys():
            dist = self.dists[self.name2idx[name]]
            val = hp_dict[name]

            if dist.is_numerical:
                val = dist.normalize_to_range(val)
            else:
                val = dist.option_to_int(val)
            normalized[name] = val

        if return_as_list:
            lst = []
            for name in self.var_names:
                if name in normalized.keys():
                    lst.append(normalized[name])
            return lst
        else:
            return normalized

    def unnormalize(self, hp_dict: Dict[str, Any]) -> Dict[str]:
        """Reverse the normalize_to_space function.

        Parameters
        ----------
        hp_dict: Dict[str, Any]
            Dictionary of normalized hyperparameter names and values.

        Returns
        ----------
        Dict[str, Any]
            Dictionary of unnormalized hyperparameter names and values.
        """
        unnormalized = {}
        for name in hp_dict.keys():
            dist = self.dists[self.name2idx[name]]
            val = hp_dict[name]

            if dist.is_numerical:
                val = dist.unnormalize(val)
            else:
                val = dist.int_to_option(val)
            unnormalized[name] = val
        return unnormalized


class Searcher(ABC):
    """
    Base searcher class.  Searchers handle hyperparameter
    configuration selection and only  have access to a) the
    hyperparameters that need to be tuned and b) the performance of
    these hyperparameters.
    """

    def __init__(self):
        """Initialize the searcher.
        """
        self.space = None

    def assign_space(self, space: Space):
        """
        Assign a space of distributions for the searcher to search over.

        Parameters
        ----------
        space: Space
            A Space object that holds the distributions to search over.
        """
        space = Space(space)
        self._check_space(space)
        self.space = space
        self.n_configs_proposed = 0

    @classmethod
    def _check_space(cls, space: Space):
        """
        Check that a particular space object is valid for the searcher.

        Parameters
        ----------
        space: Space
            A Space object that holds the distributions to search over.

        Raises
        ----------
        ValueError
            If invalid space object is passed to searcher.
        """
        pass

    def propose_new_params(self, **kwargs) -> Tuple[int, Dict[str, Any]]:
        """
        Proposes a new hyperparameter configuration along with unique
        id based on internal searching algorithm.

        Returns
        ----------
        int
            The unique id of the hyperparameters
        Dict[str, Any]
            The hyperparameters proposed by the algorithm
        """
        if self.space is None:
            raise ValueError('Space has not been assigned to searcher.')
        params_id = self.n_configs_proposed
        params = self._propose_new_params(**kwargs)
        if params is None:
            return None, None
        else:
            self.n_configs_proposed += 1
            return params_id, params

    @abstractmethod
    def _propose_new_params(self, **kwargs) -> Dict[str, Any]:
        """
        Proposes a new hyperparameter configuration based on
        internal searching algorithm.  All subclassses must implement
        this method.

        Returns
        ----------
        Dict[str, Any]
            The hyperparameters proposed by the algorithm
        """
        pass

    @abstractmethod
    def register_results(self, results: Dict[int, float], **kwargs):
        """
        Records performance of hyperparameter configurations and adapts
        the logic of internal searching algorithm.

        Parameters
        ----------
        results: Dict[int, float]
            The dictionary of results mapping parameter id to values
        """
        pass


class ModelBasedSearcher(Searcher):
    """Base class for model-based searching algorithms.

    Model-based searching algorithms iteratively build a model to
    determine which hyperparameter configurations to propose.
    """
    def __init__(self, min_configs_in_model: int):
        """Initializes model-based searcher.

        Parameters
        ----------
        min_points_in_model: int
            Minimum number of points to collect before model building.
        """
        super().__init__()
        self.min_configs_in_model = min_configs_in_model
        self.data = {}

    @property
    def n_configs_in_model(self) -> int:
        """Number of recorded points by searcher.

        Returns
        ----------
        int
            Number of recorded points by searcher.
        """
        return sum([1 for datum in self.data.values() if datum['result'] is not None])

    @property
    def has_enough_configs_for_model(self) -> bool:
        """
        Check if the searcher has enough points to build a model.

        Returns
        ----------
        bool
            If the model has more points than the minimum required.
        """
        return self.n_configs_in_model >= self.min_configs_in_model

    def _propose_new_params(self, **kwargs) -> Dict[str, Any]:
        """
        Propose a new hyperparameter configuration.
        function and applies the transforms from the space object.
        Note that model building happens in the distribution space (not
        the transformed space).

        Returns
        ----------
        Dict[str, Any]
            The configuration proposed by the searcher.
        """
        params_in_model_space = self._propose_new_params_in_model_space(**kwargs)
        params_id = self.n_configs_proposed
        self.data[params_id] = {'params_in_model_space': params_in_model_space,
                                'result': None}

        params = self._apply_transform(params_in_model_space)
        return params

    @abstractmethod
    def _propose_new_params_in_model_space(self, **kwargs) -> Dict[str, Any]:
        """
        Generates samples in the distribution space (also called model
        space).  Subclasses must override this method.

        Returns
        ----------
        Dict[str, Any]
            The configuration proposed by the model.
        """
        if not self.has_enough_configs_for_model:
            return self.space.sample_raw_dist()

    @abstractmethod
    def _apply_transform(self, params_in_model_space) -> Dict[str, Any]:
        """
        Applies the transforms to the parameters proposed in model
        space.  Subclasses must override this method.

        Parameters
        ----------
        params_in_model_space: Dict[str, Any]
            Dictionary of parameters in model space.

        Returns
        ----------
        Dict[str, Any]
            Dictionary of parameters after transforms have been applied.

        """
        pass

    def register_results(self, results: Dict[int, float], **kwargs):
        """
        Records performance of hyperparameter configurations, storing
        these as a dataset for internal model building.

        Parameters
        ----------
        results: Dict[int, float]
            The dictionary of results mapping parameter id to values.
        """
        for params_id, result in results.items():
            self.data[params_id]['result'] = result
