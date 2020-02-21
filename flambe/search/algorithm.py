from abc import abstractmethod
from typing import Dict, Optional

from flambe.compile import Registrable, YAMLLoadType
from flambe.search.distribution import Distribution
from flambe.search.trial import Trial
from flambe.search.searcher import GridSearcher, RandomSearcher,\
    BayesOptGPSearcher, BayesOptKDESearcher
from flambe.search.scheduler import Scheduler, BlackBoxScheduler, HyperBandScheduler


class Algorithm(Registrable):
    """Interface for hyperparameter search algorithms."""

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.KWARGS

    @abstractmethod
    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Mapping from option name to values.

        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        -------
        boool
            ``True`` if the algorithm has terminated.

        """
        pass

    @abstractmethod
    def update(self, trials: Dict[str, Trial], maximum: int) -> Dict[str, Trial]:
        """Update the state of the search.

        Parameters
        ----------
        trials: Dict[str, Trial]
            Mapping from trial id to Trial object.
        maximum: int
            The maximum number of new trials to fetch.

        Returns
        -------
        Dict[str, Trial]
            Mapping from trial id to Trial object, after updates.

        """
        pass


class BaseAlgorithm(Algorithm):
    """Base implementation of a hyperparameter search algorithm."""

    def __init__(self, space: Optional[Dict[str, Distribution]] = None) -> None:
        """Initialize a BaseAlgorithm.

        Parameters
        ----------
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        self.scheduler: Optional[Scheduler] = None

        self._initialized = False
        if space is not None:
            self.initialize(space)

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            A space of hyperparameters to search over.

        """
        if self._initialized:
            raise ValueError("Algorithm was already initialized.")
        self._initialized = True

    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials."""
        if not self._initialized:
            raise ValueError('Algorithm has not been initialized with search space.')

        return self.scheduler.is_done()  # type: ignore

    def update(self, trials: Dict[str, Trial], maximum: int) -> Dict[str, Trial]:
        """Update the state of the search."""
        if not self._initialized:
            raise ValueError('Algorithm has not been initialized with search space.')

        trials = self.scheduler.update_trials(trials)  # type: ignore
        trials = self.scheduler.release_trials(maximum, trials)   # type: ignore
        return trials


class GridSearch(BaseAlgorithm):
    """The grid search algorithm.

    Simply runs the cross product of all search options. Only
    supports Choice objects as distributions, and the probabilities
    will be ignored.

    """

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            A space of hyperparameters to search over.

        """
        super().initialize(space)

        searcher = GridSearcher(space)
        self.scheduler = BlackBoxScheduler(
            searcher=searcher,
            trial_budget=float('inf'),  # type: ignore
        )


class RandomSearch(BaseAlgorithm):

    def __init__(self,
                 trial_budget: int,
                 seed: Optional[int] = None,
                 space: Optional[Dict[str, Distribution]] = None):
        """The random search algorithm.

        Samples randomly from the search space.

        Parameters
        ----------
        trial_budget: int
            The number of trials to sample before ending.
        seed: int, optional
            A seed for the random search.
        space : Dict[str, Options], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        super().__init__(space=space)
        self.trial_budget = trial_budget
        self.seed = seed

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            A space of hyperparameters to search over.

        """
        super().initialize(space)

        searcher = RandomSearcher(space, seed=self.seed)
        self.scheduler = BlackBoxScheduler(
            searcher=searcher,
            trial_budget=self.trial_budget
        )


class BayesOptGP(BaseAlgorithm):

    def __init__(self,
                 trial_budget: int,
                 min_configs_in_model: int = 1,
                 aq_func: str = "ei",
                 kappa: float = 2.5,
                 xi: float = 0.0,
                 seed: int = None,
                 space: Optional[Dict[str, Distribution]] = None):
        """The BayesOpt algorithm with Gaussian Processes.

        Learns how to search over the space with gaussian processes.

        Parameters
        ----------
        trial_budget : int
            A total budget of trials.
        min_configs_in_model : int, optional
            The minimum number of trials to run before fetching from
            the algorithm, by default None.
        aq_func: str
            Aquisition function type; possibilities: "ei", "ucb", "poi".
            Default ``'ei'``.
        kappa: float
            Acquisition function parameter `kappa`. Default ``2.5``.
        xi: float
            Acquisition function parameter `xi`. Default ``0``.
        seed : int, optional
            A seed for the searcher.
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.e

        """
        super().__init__(space=space)

        self.trial_budget = trial_budget
        self.seed = seed
        self.aq_func = aq_func
        self.kappa = kappa
        self.xi = xi
        self.min_configs_in_model = min_configs_in_model

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            A space of hyperparameters to search over.

        """
        super().initialize(space)

        searcher = BayesOptGPSearcher(
            space=space,
            min_configs_in_model=self.min_configs_in_model,
            aq_func=self.aq_func,
            kappa=self.kappa,
            xi=self.xi,
            seed=self.seed
        )
        self.scheduler = BlackBoxScheduler(
            searcher=searcher,
            trial_budget=self.trial_budget
        )


class BayesOptKDE(BaseAlgorithm):

    def __init__(self,
                 trial_budget,
                 min_configs_per_model: Optional[int] = None,
                 top_n_frac: float = 0.15,
                 num_samples: int = 64,
                 random_fraction: float = 0.33,
                 bandwidth_factor: float = 3,
                 min_bandwidth: float = 1e-3,
                 space: Optional[Dict[str, Distribution]] = None):
        """The BayesOpt algorithm with KDE.

        Learns how to search over the space with KDE.

        Parameters
        ----------
        trial_budget : int
            A total budget of trials.
        min_configs_per_model: int, optional
            Minimum number of points per model before model building.
            Note that there are two models for this searcher.
        top_n_frac: float
            Fraction of points to use for building the "good" model.
            Default ``0.15``.
        num_samples: int
            Number of samples for optimizing the acquisition function.
            Default ``64``.
        random_fraction: float
            Fraction of time to return a randomly generated sample.
            Default ``0.33``.
        bandwidth_factor: float
            Algorithm parameter for encouraging exploration.
            Default ``3``.
        min_bandwidth: float
            Minimum bandwidth. Default ``1e-3``.
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        super().__init__(space=space)

        self.trial_budget = trial_budget
        self.top_n_frac = top_n_frac
        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.bandwidth_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.min_configs_per_model = min_configs_per_model

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            A space of hyperparameters to search over.

        """
        super().initialize(space)

        searcher = BayesOptKDESearcher(
            space=space,
            min_configs_per_model=self.min_configs_per_model,
            top_n_frac=self.top_n_frac,
            num_samples=self.num_samples,
            random_fraction=self.random_fraction,
            bandwidth_factor=self.bandwidth_factor,
            min_bandwidth=self.min_bandwidth
        )
        self.scheduler = BlackBoxScheduler(
            searcher=searcher,
            trial_budget=self.trial_budget
        )


class Hyperband(BaseAlgorithm):

    def __init__(self,
                 step_budget: int,
                 max_steps: int,
                 min_steps: int = 1,
                 drop_rate: float = 3,
                 seed: Optional[int] = None,
                 space: Optional[Dict[str, Distribution]] = None):
        """The Hyperband algorithm.

        Paper: https://arxiv.org/abs/1603.06560

        Parameters
        ----------
        step_budget : int
            Total number of steps to budget for the search.
        max_steps : int
            The maximum number of steps to run per trial.
        min_steps : int, optional
            The minimum number of steps to run per trial, by default 1
        drop_rate: float, optional
            The rate at which trials are dropped between rounds of the
            HyperBand algorithm.  A higher drop rate means that the
            algorithm will be more exploitative than exploratory.
            Default ``3``.
        seed : Optional[int], optional
            A seed for the search, by default None
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        super().__init__(space=space)
        self.step_budget = step_budget
        self.seed = seed
        self.drop_rate = drop_rate
        self.max_steps = max_steps
        self.min_steps = min_steps

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            A space of hyperparameters to search over..

        """
        super().initialize(space)

        searcher = RandomSearcher(space, seed=self.seed)
        self.scheduler = HyperBandScheduler(
            searcher=searcher,
            step_budget=self.step_budget,
            drop_rate=self.drop_rate,
            max_steps=self.max_steps,
            min_steps=self.min_steps
        )


class BOHB(BaseAlgorithm):

    def __init__(self,
                 step_budget: int,
                 max_steps: int = 1,
                 min_steps: int = 1,
                 drop_rate: int = 3,
                 min_configs_per_model: Optional[int] = None,
                 top_n_frac: float = 0.15,
                 num_samples: int = 64,
                 random_fraction: float = 1 / 3,
                 bandwidth_factor: int = 3,
                 min_bandwidth: float = 1e-3,
                 seed: Optional[int] = None,
                 space: Optional[Dict[str, Distribution]] = None):
        """The BOHB algorithm.

        Uses bayesopt for selecting new trials, and hyperband to
        schedule the trials given a budget.

        Paper: https://arxiv.org/abs/1807.01774.

        Parameters
        ----------
        step_budget : int
            Total number of steps to budget for the search.
        max_steps : int
            The maximum number of steps to run per trial.
        min_steps : int, optional
            The minimum number of steps to run per trial, by default 1
        drop_rate: float, optional
            The rate at which trials are dropped between rounds of the
            HyperBand algorithm.  A higher drop rate means that the
            algorithm will be more exploitative than exploratory.
            Default ``3``.
        min_configs_per_model: int, optional
            Minimum number of points per model before model building.
            Note that there are two models for this searcher.
        top_n_frac: float
            Fraction of points to use for building the "good" model.
            Default ``0.15``.
        num_samples: int
            Number of samples for optimizing the acquisition function.
            Default ``64``.
        random_fraction: float
            Fraction of time to return a randomly generated sample.
            Default ``0.33``.
        bandwidth_factor: float
            Algorithm parameter for encouraging exploration.
            Default ``3``.
        min_bandwidth: float
            Minimum bandwidth. Default ``1e-3``.
        seed : Optional[int], optional
            A seed for the search, by default None
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        super().__init__(space=space)

        self.seed = seed
        self.step_budget = step_budget
        self.drop_rate = drop_rate
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.top_n_frac = top_n_frac
        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.bandwidth_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.min_configs_per_model = min_configs_per_model

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            A space of hyperparameters to search over..

        """
        super().initialize(space)

        searcher = BayesOptKDESearcher(
            space=space,
            min_configs_per_model=self.min_configs_per_model,
            top_n_frac=self.top_n_frac,
            num_samples=self.num_samples,
            random_fraction=self.random_fraction,
            bandwidth_factor=self.bandwidth_factor,
            min_bandwidth=self.min_bandwidth
        )
        self.scheduler = HyperBandScheduler(
            searcher=searcher,
            step_budget=self.step_budget,
            drop_rate=self.drop_rate,
            max_steps=self.max_steps,
            min_steps=self.min_steps
        )
