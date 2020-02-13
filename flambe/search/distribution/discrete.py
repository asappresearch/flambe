from typing import List, Optional, Callable, Any, Dict, Union
from numbers import Number

import numpy as np

from flambe.search.distribution.numerical import Numerical


class Discrete(Numerical):
    """Base discrete variable that inherbits from numerical class."""

    def __init__(self,
                 options: List[Number],
                 probs: Optional[List[float]] = None,
                 dist_params: Optional[Dict[str, Any]] = None,
                 transform: Optional[Union[str, Callable]] = None) -> None:
        """Initialize the distribution.

        Parameters
        ----------
        options : List[Number]
            List of options, where each option is a number.
        probs : List[float], optional
            A corresponding multinomial distribution, by default None.
        dist_params: Dict[str, Any], optional
            Dictionary denoting the parameters of the distribution.
        transform: Union[str, Callable], optional
            String or custom function for transforming
            the variable after sampling.  Possible string options
            include: ['pow2', 'pow10', 'exp'].

        """
        super().__init__(
            var_min=np.min(options),
            var_max=np.max(options),
            transform=transform,
            dist_params=dist_params
        )

        self.named_options = [(str(opt), opt) for opt in options]
        self.options = np.array(options)
        self.n_options = len(options)
        if probs is None:
            self.probs = np.array([1 / self.n_options] * self.n_options)
        else:
            self.probs = np.array(probs)

    def round_to_options(self, val: float):
        """Find the closest numerical choice for a given value.

        Parameters
        ----------
        float
            The given value to round.

        Returns
        -------
        float
            The rounded options.

        """
        argmin = np.argmin([abs(val - c) for c in self.options])
        return self.options[argmin]

    def sample_raw_dist(self) -> float:
        """Sample from the distribution.

        Returns
        -------
        float
            The rounded options.

        """
        return np.random.choice(self.options, p=self.probs)


class QUniform(Discrete, tag_override="~q"):
    """Quantized uniform variable class."""

    def __init__(self,
                 low: int,
                 high: int,
                 n: int,
                 transform: Optional[Union[str, Callable]] = None) -> None:
        """Initialize the distribution.

        Parameters
        ----------
        low: int
            Minimum value.
        high: int
            Maximum value.
        n: int
            Number of evenly-spaced choices between [low, high].
        transform: Union[str, Callable], optional
            String or custom function for transforming
            the variable after sampling.  Possible string options
            include: ['pow2', 'pow10', 'exp'].

        """
        if low > high:
            raise ValueError('Parameter `low`={} cannot be greater '
                             'than parameter `high`={}.'.format(low, high))

        super().__init__(options=np.linspace(low, high, n),
                         probs=np.array([1 / n] * n),
                         dist_params={'low': low, 'high': high, 'n': n},
                         transform=transform)
