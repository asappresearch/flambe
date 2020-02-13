from abc import abstractmethod
import inspect
from typing import Optional, Dict, Callable, Union, Any, Tuple

import numpy as np

from flambe.search.distribution.distribution import Distribution


class Numerical(Distribution):
    """Base numerical variable."""

    def __init__(self,
                 var_min: float,
                 var_max: float,
                 transform: Optional[Union[str, Callable]] = None,
                 dist_params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize a numerical distribution.

        Parameters
        ----------
        var_min: float
            Minimum bound of variable.
        var_max: float
            Maximum bound of variable.
        transform: Union[str, Callable]
            String or custom function for transforming
            the variable after sampling.  Possible string options
            include: ['pow2', 'pow10', 'exp'].
        dist_params: Dict[str, Any], optional
            Dictionary denoting the parameters of the distribution.

        """
        if dist_params is None:
            dist_params = dict()

        self.var_min = var_min
        self.var_max = var_max

        if transform is None:
            self.transform_fn = lambda x: x
        elif transform == 'pow2':
            self.transform_fn = lambda x: 2**x
        elif transform == 'pow10':
            self.transform_fn = lambda x: 10**x
        elif transform == 'exp':
            self.transform_fn = lambda x: np.exp(x)
        elif isinstance(transform, str):
            raise ValueError(f"Unsupported transform {transform}.")
        elif inspect.isfunction(transform):
            self.transform_fn = transform
        else:
            raise ValueError(f"transform should be a string or a function.")

        self.dist_params = dist_params

    @abstractmethod
    def sample_raw_dist(self) -> float:
        """Sample from the raw distribution.

        Returns
        -------
        float
            The sampled value.

        """
        pass

    def named_sample_raw_dist(self) -> Tuple[str, float]:
        """Sample from the raw distribution, with a name.

        Returns
        -------
        float
            The sampled value.

        """
        samp = self.sample_raw_dist()
        return str(samp), samp

    def sample(self) -> float:
        """Sample from the transformed distribution.

        Returns
        -------
        float
            The sampled value.

        """
        samp = self.sample_raw_dist()
        return self.transform_fn(samp)

    def normalize_to_range(self, val: float) -> float:
        """Standardize the variable by its min and max bounds.

        Parameters
        ----------
        val: float
            The value of the variable.

        Returns
        -------
        float
            The normalized value.

        """
        if val < self.var_min or val > self.var_max:
            raise ValueError('Given value outside of range!')
        else:
            return (val - self.var_min) / (self.var_max - self.var_min)

    def unnormalize(self, val: float) -> float:
        """Reverse the normalize_to_range function.

        Parameters
        ----------
        val: float
            The value of the variable.

        Returns
        -------
        float
            The unnormalized value.

        """
        if val < 0 or val > 1:
            raise ValueError('Normalized value must be between [0, 1].')
        else:
            return (self.var_max - self.var_min) * val + self.var_min
