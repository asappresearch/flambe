from typing import Any, Dict, Optional, List

import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from scipy.stats import truncnorm
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from flambe.search.distribution import (Choice, Continuous, Discrete,
                                        Distribution, Numerical)
from flambe.search.searcher.searcher import ModelBasedSearcher, Space


class BayesOptGPSearcher(ModelBasedSearcher):
    """Bayesian optimization with Gaussian Process searcher.

    Bayesian optimization search models the hyperparameter space
    with a Gaussian process surrogate function and uses Bayesian
    principles to iteratively propose configurations that trade
    off exploration and exploitation.

    Wraps around BayesianOptimization library:
    https://github.com/fmfn/BayesianOptimization.

    """

    def __init__(self,
                 space: Dict[str, Distribution],
                 min_configs_in_model: int = 1,
                 aq_func: str = "ei",
                 kappa: float = 2.5,
                 xi: float = 0.0,
                 seed: Optional[int] = None):
        """Initializes Bayesian optimization GP searcher.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Dictionary mapping variable names to their initial
            distributions.
        min_points_in_model: int
            Minimum number of points before model-based searching
            starts.
        aq_func: str
            Aquisition function type; possibilities: "ei", "ucb", "poi".
        kappa: float
            Acquisition function parameter `kappa`.
        xi: float
            Acquisition function parameter `xi`.
        seed: Optional[int]
            Seed for the searcher.

        """
        super().__init__(space, min_configs_in_model)

        self.aq_func = aq_func
        self.kappa = kappa
        self.xi = xi
        self.seed = seed

        # Initialize BayesOpt library objects
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds=self.space.var_bounds,
            verbose=0,
            random_state=self.seed
        )
        self.utility = UtilityFunction(self.aq_func, self.kappa, self.xi)

    @classmethod
    def check_space(cls, space: Space):
        """
        Check if the space is valid for this algorithm, which only
        accepts numerical (continuous and discrete) distributions.

        Parameters
        ----------
        space: Space
            The search space to check.

        """
        dists = space.dists.values()
        if any(not isinstance(dist, Numerical) for dist in dists):
            raise ValueError('For grid search, all dimensions \
                             must be `continuous` or `discrete`!')

    def _propose_new_params_in_model_space(self) -> Optional[Dict[str, Any]]:
        """Propose a new hyperparameter configuration.

        Return the config a a dictionary, along with unique
        id based on internal searching algorithm.

        Returns
        -------
        Dict[str, Any], optional
            The configuration proposed by the searcher.

        """
        # Ensure there are no proposed trials awaiting result
        if any(k not in self.results for k in self.params):
            raise ValueError('Gaussian Process searcher cannot propose new\
                             configuration with missing result.')

        rand_samp = self.space.sample()
        if rand_samp is None:
            return self.optimizer.suggest(self.utility)
        else:
            return rand_samp

    def _apply_transform(self, params_in_model_space: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform to parameters in model space.

        Parameters
        ----------
        params_in_model_space: Dict[str, Any]
            The configuration with parameters in model space.

        Returns
        ----------
        Dict[str, Any]
            The configuration with parameters in transformed space.

        """
        params_in_raw_dist_space = self.space.round_to_space(params_in_model_space)
        params = self.space.apply_transform(params_in_raw_dist_space)
        return params

    def register_results(self, results: Dict[str, float]):
        """Record the new results and fit a GP.

        Parameters
        ----------
        results: Dict[str, float]
            Dictionary of hyperparameter ids and results to record.

        """
        super().register_results(results)

        for params_id in results.keys():
            self.optimizer.register(params=self.params[params_id], target=self.results[params_id])


class BayesOptKDESearcher(ModelBasedSearcher):
    """Bayesian optimization with kernel density estimation searcher.

    Adapted from:
    https://github.com/automl/HpBandSter/blob/master/hpbandster/optimizers/config_generators/bohb.py

    Mathematical details can be found in:
    https://bookdown.org/egarpor/NP-UC3M/kre-ii-multmix.html

    """

    def __init__(self,
                 space: Dict[str, Distribution],
                 min_configs_per_model: Optional[int] = None,
                 top_n_frac: float = 0.15,
                 num_samples: int = 64,
                 random_fraction: float = 1 / 3,
                 bandwidth_factor: float = 3,
                 min_bandwidth: float = 1e-3):
        """Initialize the Bayesian optimization KDE searcher.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Dictionary mapping variable names to their initial
            distributions.
        min_configs_per_model: int
            Minimum number of points per model before model building.
            Note that there are two models for this searcher.
        top_n_frac: float
            Fraction of points to use for building the "good" model.
        num_samples: int
            Number of samples for optimizing the acquisition function.
        random_fraction: float
            Fraction of time to return a randomly generated sample.
        bandwidth_factor: float
            Algorithm parameter for encouraging exploration.
        min_bandwidth: float
            Minimum bandwidth.

        """
        if min_configs_per_model is None:
            min_configs_per_model = len(space) + 1
            min_configs_in_model = min_configs_per_model + 2

        if min_configs_per_model < len(space) + 1:
            raise ValueError('Parameter min_points_in_model cannot be \
                             less than one plus the number of hyperparameters.')

        super().__init__(space, min_configs_in_model)

        self.min_configs_per_model = min_configs_per_model
        self.top_n_frac = top_n_frac
        self.bw_factor = bandwidth_factor
        self.min_bw = min_bandwidth

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        self.good_kde: Optional[KDEMultivariate] = None
        self.bad_kde: Optional[KDEMultivariate] = None

    def check_space(self, space: Space):
        """Check if the space is valid for this algorithm.

        Does not allow discrete distributions.

        Parameters
        ----------
        space: Space
            A Space object that holds the distributions to search over.

        """
        if np.any([isinstance(x, Discrete) for x in space.dists.values()]):
            raise ValueError('BayesOpt KDE Searcher does not support \
                             distributions with `var_type=discrete`!')

    def _propose_new_params_in_model_space(self) -> Optional[Dict[str, Any]]:
        """Propose a new hyperparameter configuration.

        Return the config a a dictionary, along with unique
        id based on internal searching algorithm.

        Returns
        -------
        Dict[str, Any], optional
            The configuration proposed by the searcher.

        """
        rand_samp = super()._propose_new_params_in_model_space()
        if rand_samp is not None:
            return self.space.normalize_to_space(rand_samp)

        if np.random.uniform() < self.random_fraction:
            return self.space.normalize_to_space(self.space.sample())

        best = np.inf
        best_hp: List[int] = []

        def func_to_min(x):
            return max(1e-32, self.bad_kde.pdf(x)) / max(self.good_kde.pdf(x), 1e-32)

        for _ in range(self.num_samples):

            # Sample from KDE
            idx = np.random.randint(0, len(self.good_kde.data))  # type: ignore
            datum = self.good_kde.data[idx]  # type: ignore
            sample_hp: List[int] = []

            for m, bw, dist in zip(datum, self.good_kde.bw, self.space.dists.values()):  # type: ignore
                bw = max(bw, self.min_bw)
                if isinstance(dist, Continuous):
                    bw = self.bw_factor * bw
                    sample_hp.append(truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))

                elif isinstance(dist, Choice):
                    bw = min(bw, 1)  # Probability cannot be greater than 1
                    if np.random.rand() < 1 - bw:
                        sample_hp.append(int(m))
                    else:
                        # technically should exclude m
                        sample_hp.append(np.random.randint(dist.n_options))
                else:
                    raise ValueError('Unknown var type!')

            val = func_to_min(sample_hp)
            if val < best:
                best = val
                best_hp = sample_hp

        best_hp_dict = {n: hp for n, hp in zip(self.space.dists.keys(), best_hp)}
        return best_hp_dict

    def _apply_transform(self, params_in_model_space: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to parameters in model space.

        Parameters
        ----------
        params_in_model_space: Dict[str, Any]
            The parameters proposed in model space.

        Returns
        ----------
        Dict[str, Any]
            The parameters after transform has been applied in model
            space.
        """
        params_in_raw_dist_space = self.space.unnormalize(params_in_model_space)
        params = self.space.apply_transform(params_in_raw_dist_space)
        return params

    def register_results(self, results: Dict[str, float]):
        """Record the new results and fit KDEs.

        Parameters
        ----------
        results: Dict[str, float]
            Dictionary of hyperparameter ids and results to record.

        """
        super().register_results(results)

        if self.has_enough_configs_for_model:

            finished = set([k for k in self.params if k in self.results])
            train_data = [list(x.values()) for k, x in self.params.items() if k in finished]
            train_results = [self.results[k] for k in finished]

            min_configs = self.min_configs_per_model
            n_good = max(min_configs, int(self.top_n_frac * min_configs))
            n_bad = max(min_configs, self.n_configs_in_model - n_good)

            idx = np.argsort(train_results)[::-1]
            good_train_data = np.array(train_data)[idx[:n_good]]
            bad_train_data = np.array(train_data)[idx[-n_bad:]]

            # Fit KDE
            kde_vartypes = ['c' if isinstance(x, Continuous) else 'u' for x in self.space.dists.values()]
            self.good_kde = KDEMultivariate(data=good_train_data,
                                            var_type=kde_vartypes,
                                            bw='normal_reference')
            self.bad_kde = KDEMultivariate(data=bad_train_data,
                                           var_type=kde_vartypes,
                                           bw='normal_reference')

            self.good_kde.bw = np.clip(self.good_kde.bw, self.min_bw, None)
            self.bad_kde.bw = np.clip(self.bad_kde.bw, self.min_bw, None)
