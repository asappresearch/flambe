from scipy.stats import truncnorm
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from bayes_opt import BayesianOptimization, UtilityFunction
import numpy as np
from typing import List, Dict, Any, Optional

from flambe.search.searcher.searcher import ModelBasedSearcher
from flambe.search.searcher.searcher import Space


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
                 min_configs_in_model: int = 1,
                 aq_func: str = "ei",
                 kappa: float = 2.5,
                 xi: float = 0.0,
                 seed: Optional[int] = None):
        """Initializes Bayesian optimization GP searcher.

        Parameters
        ----------
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

        super().__init__(min_configs_in_model)

        self.aq_func = aq_func
        self.kappa = kappa
        self.xi = xi
        self.seed = seed

    @classmethod
    def _check_space(cls, space: Space):
        """
        Check if the space is valid for this algorithm, which only
        accepts numerical (continuous and discrete) distributions.

        Parameters
        ----------
        space: Space
            The search space to check.
        """
        if np.any(np.array(space.var_types) == 'choice'):
            raise ValueError('For grid search, all dimensions \
                             must be `continuous` or `discrete`!')

    def assign_space(self, space):
        """Assign a search space to the searcher.

        Parameters
        ----------
        space: Space
            The search space to assign.
        """
        super().assign_space(space)

        # Initialize BayesOpt library objects
        self.optimizer = BayesianOptimization(
            f=None,
            pbounds={n: b for n, b in zip(self.space.var_names, self.space.var_bounds)},
            verbose=0,
            random_state=self.seed
        )
        self.utility = UtilityFunction(self.aq_func, self.kappa, self.xi)

    def _propose_new_params_in_model_space(self, **kwargs) -> Dict[Any]:
        """
        Use the GP and acquisition function to propose a
        new (unrounded) hyperparameter configuration.

        Returns
        ----------
        Dict[Any]
            The configuration proposed by the model.
        """
        # Ensure there are no proposed trials awaiting result
        if np.any([x['result'] is None for x in self.data.values()]):
            raise ValueError('Gaussian Process searcher cannot propose new\
                             configuration with missing result.')

        rand_samp = self.space.sample_raw_dist(**kwargs)
        if rand_samp is None:
            return self.optimizer.suggest(self.utility)
        else:
            return rand_samp

    def _apply_transform(self, params_in_model_space: Dict[Any]) -> Dict[Any]:
        """Apply transform to parameters in model space.

        Parameters
        ----------
        params_in_model_space: Dict[Any]
            The configuration with parameters in model space.

        Returns
        ----------
        Dict[Any]
            The configuration with parameters in transformed space.
        """
        params_in_raw_dist_space = self.space.round_to_space(params_in_model_space)
        params = self.space.apply_transform(params_in_raw_dist_space)
        return params

    def register_results(self, results: Dict[int, float], **kwargs):
        """Record the new results and fit a GP.

        Parameters
        ----------
        results: Dict[int, float]
            Dictionary of hyperparameter ids and results to record.
        """
        super().register_results(results, **kwargs)

        for params_id in results.keys():
            self.optimizer.register(params=self.data[params_id]['params_in_model_space'],
                                    target=self.data[params_id]['result'])


class BayesOptKDESearcher(ModelBasedSearcher):
    """Bayesian optimization with kernel density estimation searcher.

    Adapted from:
    https://github.com/automl/HpBandSter/blob/master/hpbandster/optimizers/config_generators/bohb.py

    Mathematical details can be found in:
    https://bookdown.org/egarpor/NP-UC3M/kre-ii-multmix.html
    """

    def __init__(self,
                 min_configs_per_model: Optional[int] = None,
                 top_n_frac: float = 0.15,
                 num_samples: int = 64,
                 random_fraction: float = 1/3,
                 bandwidth_factor: float = 3,
                 min_bandwidth: float = 1e-3):
        """Initialize the Bayesian optimization KDE searcher.

        Parameters
        ----------
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

        self.top_n_frac = top_n_frac
        self.bw_factor = bandwidth_factor
        self.min_bw = min_bandwidth

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        self.good_kde = None
        self.bad_kde = None

        self.min_configs_per_model = min_configs_per_model
        if min_configs_per_model:
            super().__init__(min_configs_per_model + 2)
        else:
            super().__init__(-1)

    def assign_space(self, space: Space):
        """Assign a search space to the searcher.

        Parameters
        ----------
        space: Space
            The search space to assign.
        """
        super().assign_space(space)

        if self.min_configs_per_model is None:
            self.min_configs_per_model = self.space.n_vars + 1
            self.min_configs_in_model = self.min_configs_per_model + 2

        if self.min_configs_per_model < self.space.n_vars + 1:
            raise ValueError('Parameter min_points_in_model cannot be \
                             less than one plus the number of hyperparameters.')

        self.kde_vartypes = [vt[0] for vt in self.space.var_types]

    @classmethod
    def _check_space(cls, space: Space):
        """
        Check if the space is valid for this algorithm, which does not
        accept discrete distributions.

        Parameters
        ----------
        space: Space
            The search space to check.
        """
        if np.any([x == 'discrete' for x in space.var_types]):
            raise ValueError('BayesOpt KDE Searcher does not support \
                             distributions with `var_type=discrete`!')

    def _propose_new_params_in_model_space(self, **kwargs) -> Dist[str, Any]:
        """
        Propose new model space hyperparameters.  The model is built
        in a normalized (untransformed) space.

        Returns
        ----------
        Dict[str, Any]
            The configuration of hyperparameters in model space.
        """

        rand_samp = super()._propose_new_params_in_model_space(**kwargs)
        if rand_samp is not None:
            return self.space.normalize_to_space(rand_samp)

        if np.random.uniform() < self.random_fraction:
            return self.space.normalize_to_space(self.space.sample_raw_dist())

        best = np.inf
        best_hp = None

        def func_to_min(x):
            return max(1e-32, self.bad_kde.pdf(x)) / max(self.good_kde.pdf(x), 1e-32)

        for _ in range(self.num_samples):

            # Sample from KDE
            idx = np.random.randint(0, len(self.good_kde.data))
            datum = self.good_kde.data[idx]
            sample_hp = []

            for m, bw, dist in zip(datum, self.good_kde.bw, self.space.dists):
                bw = max(bw, self.min_bw)
                if dist.var_type == 'continuous':
                    bw = self.bw_factor * bw
                    sample_hp.append(truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))

                elif dist.var_type == 'choice':
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

        best_hp_dict = {n: hp for n, hp in zip(self.space.var_names, best_hp)}
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

    def register_results(self, results: Dict[int, float], **kwargs):
        """Record the new results and fit KDEs.

        Parameters
        ----------
        results: Dict[int, float]
            Dictionary of hyperparameter ids and results to record.
        """

        super().register_results(results, **kwargs)

        if self.has_enough_configs_for_model:

            train_data = [list(x['params_in_model_space'].values()) for x in self.data.values()]
            train_results = [x['result'] for x in self.data.values()]

            min_configs = self.min_configs_per_model
            n_good = max(min_configs, int(self.top_n_frac * min_configs))
            n_bad = max(min_configs, self.n_configs_in_model - n_good)

            idx = np.argsort(train_results)[::-1]
            good_train_data = np.array(train_data)[idx[:n_good]]
            bad_train_data = np.array(train_data)[idx[-n_bad:]]

            # Fit KDE
            self.good_kde = KDEMultivariate(data=good_train_data,
                                            var_type=self.kde_vartypes,
                                            bw='normal_reference')
            self.bad_kde = KDEMultivariate(data=bad_train_data,
                                           var_type=self.kde_vartypes,
                                           bw='normal_reference')

            self.good_kde.bw = np.clip(self.good_kde.bw, self.min_bw, None)
            self.bad_kde.bw = np.clip(self.bad_kde.bw, self.min_bw, None)
