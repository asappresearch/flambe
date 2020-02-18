from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import numpy as np

from flambe.search.distribution import Distribution
from flambe.search.distribution.numerical import Numerical
from flambe.search.distribution.discrete import Discrete
from flambe.search.distribution.choice import Choice


def generate_name(parameters: Any, prefix: str = '', delimiter='|') -> str:
    """Generate a name for this parameter variant.

    Parameters
    ----------
    parameters: Any
        Parameter object to parse. Can be a sequence, mapping, or
        any other Python object.
    prefix: str, optional
        A prefix to add to the generated name.

    Returns
    -------
    str
        A string name to represent the variant when dumping results.

    """
    def helper(params):
        if isinstance(params, (list, tuple, set)):
            name = ",".join([helper(param) for param in params])
            name = f"[{name}]"
        elif isinstance(params, dict):
            name = delimiter
            for param, value in params.items():
                if isinstance(value, dict):
                    name += helper({f'{param}.{k}': v for k, v in value.items()})[1:]
                else:
                    name += f'{param}={helper(value)}{delimiter}'
        else:
            name = str(params)

        return name

    return prefix + helper(parameters).strip(delimiter)


class Space(object):
    """Search space object.

    Manages the various dimensions that the hyperparameters are in.
    This object is used internally by the search aglrothms and should
    not be used directly.

    """

    def __init__(self, distributions: Dict[str, Distribution]):
        """Initialize the search space with a set of distributions.

        Parameters
        ----------
        distributions: Dict[str, Distribution]
            Dictionary mapping variable names to their initial
            distributions.

        """
        self.dists = distributions
        self.var_bounds = dict()

        for name, dist in distributions.items():
            if isinstance(dist, Numerical):
                self.var_bounds[name] = (dist.var_min, dist.var_max)
            else:
                self.var_bounds[name] = (np.nan, np.nan)

    def sample(self) -> Dict[str, Any]:
        """Sample from the search space.

        Returns
        -------
        Dict[str, Tuple[str, Any]]
            A sample from each distribution of the searcher.

        """
        samp: Dict[str, Any] = dict()
        for name, dist in self.dists.items():
            if isinstance(dist, Numerical):
                s = dist.sample_raw_dist()
            else:
                s = dist.sample()  # type: ignore
            samp[name] = s
        return samp

    def apply_transform(self, samp: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms to a sample drawn from the space.

        Parameters
        ----------
        samp: Dict[str, Any]
            Dictionary of hyperparameter names and values.

        Returns
        ----------
        Dict[str, Any]
            Dictionary of hyperparameter names and transformed values.

        """
        transformed = dict()
        for name, value in samp.items():
            dist = self.dists[name]
            if isinstance(dist, Numerical):
                value = dist.transform_fn(value)
            transformed[name] = value
        return transformed

    def round_to_space(self, hp_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Round hyperparameters to possible choices for the variables.

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
        for name, value in hp_dict.items():
            dist = self.dists[name]
            if isinstance(dist, Discrete):
                value = dist.round_to_options(value)
            rounded[name] = value
        return rounded

    def normalize_to_space(self, hp_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize hyperparameters.

        Convert to [0, 1] based on range for numerical variables and
        convert them to integers for categorical variables.

        Parameters
        ----------
        hp_dict: Dict[str, Any]
            Dictionary of hyperparameter names and values.

        Returns
        ----------
        Dict[str, Any]
            Dictionary of hyperparameter names and values.

        """
        normalized = dict()
        for name, value in hp_dict.items():
            dist = self.dists[name]
            if isinstance(dist, Numerical):
                value = dist.normalize_to_range(value)
            elif isinstance(dist, Choice):
                value = dist.option_to_int(value)
            normalized[name] = value

        return normalized

    def unnormalize(self, hp_dict: Dict[str, Any]) -> Dict[str, Any]:
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
        unnormalized = dict()

        for name, value in hp_dict.items():
            dist = self.dists[name]
            if isinstance(dist, Numerical):
                value = dist.unnormalize(value)
            elif isinstance(dist, Choice):
                value = dist.int_to_option(value)
            unnormalized[name] = value

        return unnormalized


class Searcher(ABC):
    """Base searcher class.

    Searchers handle hyperparameter configuration selection and only
    have access to a) the hyperparameters that need to be tuned and
    b) the performance of these hyperparameters.

    """

    def __init__(self, space: Dict[str, Distribution]) -> None:
        """Assign a space of distributions to search over.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Dictionary mapping variable names to their initial
            distributions.

        """
        space = Space(space)
        self.check_space(space)
        self.space = space
        self.params: Dict[str, Dict[str, Tuple[str, Any]]] = dict()
        self.results: Dict[str, Optional[float]] = dict()

    def propose_new_params(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Propose a new hyperparameter configuration.

        Sample and applies the transforms from the space object.
        Note that model building happens in the distribution space (not
        the transformed space).

        Returns
        -------
        str, optional
            A name for this parameter config
        Dict[str, Any], optional
            The configuration proposed by the searcher.

        """
        params = self._propose_new_params_in_model_space()
        if params is None:
            return None
        else:
            # Fetch names from the distributions
            pre_transform_params = params
            params = self._apply_transform(dict(params))
            var_names = {k: self.space.dists[k].name(v) for k, v in params.items()}
            name = generate_name(var_names)
            name = name if name else '0'
            self.params[name] = pre_transform_params
            return name, params

    def register_results(self, results: Dict[str, float]):
        """Records performance of hyperparameter configurations.

        Adapts the logic of internal searching algorithm.

        Parameters
        ----------
        results: Dict[str, float]
            The dictionary of results mapping parameter id to values

        """
        for name, result in results.items():
            self.results[name] = result

    def _apply_transform(self, params_in_model_space: Dict[str, Any]) -> Dict[str, Any]:
        """Applies the transforms to the parameters proposed in model
        space. Subclasses must override this method.

        Parameters
        ----------
        params_in_model_space: Dict[str, Any]
            Dictionary of parameters in model space.

        Returns
        ----------
        Dict[str, Any]
            Dictionary of parameters after transforms have been applied.

        """
        return params_in_model_space

    @abstractmethod
    def check_space(self, space: Space):
        """Check that a particular space is valid for the searcher.

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

    @abstractmethod
    def _propose_new_params_in_model_space(self) -> Optional[Dict[str, Any]]:
        """Propose a new hyperparameter configuration.

        Return the config a a dictionary, along with unique
        id based on internal searching algorithm.

        Returns
        -------
        Dict[str, Any], optional
            The configuration proposed by the searcher.

        """
        pass


class ModelBasedSearcher(Searcher):
    """Base class for model-based searching algorithms.

    Model-based searching algorithms iteratively build a model to
    determine which hyperparameter configurations to propose.

    """

    def __init__(self, space: Dict[str, Distribution], min_configs_in_model: int) -> None:
        """Initializes model-based searcher.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Dictionary mapping variable names to their initial
            distributions.
        min_points_in_model: int
            Minimum number of points to collect before model building.

        """
        super().__init__(space)
        self.min_configs_in_model = min_configs_in_model

    @property
    def n_configs_in_model(self) -> int:
        """Number of recorded points by searcher so far.

        Returns
        -------
        int
            Number of recorded points by searcher.

        """
        return len(self.results)

    @property
    def has_enough_configs_for_model(self) -> bool:
        """Check if the searcher has enough points to build a model.

        Returns
        -------
        bool
            If the model has more points than the minimum required.

        """
        return self.n_configs_in_model >= self.min_configs_in_model

    def _propose_new_params_in_model_space(self) -> Optional[Dict[str, Any]]:
        """Generates samples in the distribution space.

        Subclasses must override this method.

        Returns
        ----------
        Dict[str, Any], optional
            The configuration proposed by the model.

        """
        if not self.has_enough_configs_for_model:
            return self.space.sample()

        return None
