from typing import Optional, Union, Callable

import numpy as np

from flambe.search.distribution.numerical import Numerical


class Continuous(Numerical):
    """Base continuous variable."""
    pass


class Uniform(Continuous, tag_override="uniform"):
    """A uniform distribution."""

    def __init__(self,
                 low: float,
                 high: float,
                 transform: Optional[Union[str, Callable]] = None) -> None:
        """Initialize the distribution.

        Parameters
        ----------
        low: float
            Minimum value of variable.
        high: float
            Maximum value of variable.
        transform: Union[str, Callable], optional
            String or custom function for transforming
            the variable after sampling.  Possible string options
            include: ['pow2', 'pow10', 'exp'].

        """
        if low > high:
            raise ValueError('Parameter `low`={} cannot be greater '
                             'than parameter `high`={}.'.format(low, high))
        super().__init__(
            var_min=low,
            var_max=high,
            transform=transform,
            dist_params={'low': low, 'high': high}
        )

    def sample_raw_dist(self) -> float:
        """Sample from the distribution.

        Returns
        -------
        float
            A sampled floating number

        """
        return np.random.uniform(**self.dist_params)


class Normal(Continuous, tag_override="normal"):
    """A normal distribution."""

    def __init__(self,
                 loc: float,
                 scale: float,
                 transform: Optional[Union[str, Callable]] = None) -> None:
        """Initialize the distribution.

        Parameters
        ----------
        loc: float
            Mean of the distribution.
        scale: float
            Standard deviation of the distribution.
        transform: Union[str, Callable], optional
            String or custom function for transforming
            the variable after sampling.  Possible string options
            include: ['pow2', 'pow10', 'exp'].

        """
        if scale < 0:
            raise ValueError('Parameter `scale`={} cannot be less than 0.'.format(scale))
        super().__init__(var_min=-np.inf, var_max=np.inf, transform=transform,
                         dist_params={'loc': loc, 'scale': scale})

    def sample_raw_dist(self) -> float:
        """Sample from the distribution.

        Returns
        -------
        float
            The sampled value.

        """
        return np.random.normal(**self.dist_params)


class Beta(Continuous, tag_override="beta"):

    def __init__(self,
                 a: int,
                 b: int,
                 transform: Optional[Union[str, Callable]] = None) -> None:
        """Initialize the distribution.

        Parameters
        ----------
        a: int
            Shape parameter of the distribution.
        b: int
            Scale parameter of the distribution.
        transform: Union[str, Callable], optional
            String or custom function for transforming
            the variable after sampling.  Possible string options
            include: ['pow2', 'pow10', 'exp'].

        """
        if a <= 0 or b <= 0:
            raise ValueError('Parameter `a`={} and `b`={} cannot be non-positive.'.format(a, b))
        super().__init__(var_min=0, var_max=1, transform=transform, dist_params={'a': a, 'b': b})

    def sample_raw_dist(self) -> float:
        """Sample from the distribution.

        Returns
        -------
        float
            The sampled value.

        """
        return np.random.beta(**self.dist_params)
