from abc import abstractmethod
from typing import Dict, Optional

from flambe.compile import RegisteredStatelessMap
from flambe.search.distribution import Distribution
from flambe.search.trial import Trial
from flambe.search.searcher import GridSearcher, RandomSearcher,\
    BayesOptGPSearcher, BayesOptKDESearcher, MultiFidSearcher
from flambe.search.scheduler import Scheduler, BlackBoxScheduler, HyperBandScheduler


class Algorithm(RegisteredStatelessMap):
    """Interface for hyperparameter search algorithms."""

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

    def __init__(self,
                 max_steps: int = 1,
                 space: Optional[Dict[str, Distribution]] = None) -> None:
        """Initialize a BaseAlgorithm.

        Parameters
        ----------
        max_steps : int, optional
            The maximum number of steps to execute, by default 1.
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        self.max_steps = max_steps
        self.scheduler: Optional[Scheduler] = None

        self._initialized = False
        if space is not None:
            self.initialize(space)

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Mapping from option name to values.

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
            Mapping from option name to values.

        """
        super().initialize(space)

        searcher = GridSearcher(space)
        self.scheduler = BlackBoxScheduler(
            searcher=searcher,
            trial_budget=float('inf'),
            max_steps=self.max_steps,

        )


class RandomSearch(BaseAlgorithm):

    def __init__(self,
                 trial_budget: int,
                 max_steps: int = 1,
                 seed: Optional[int] = None,
                 space: Optional[Dict[str, Distribution]] = None):
        """The random search algorithm.

        Samples randomly from the search space.

        Parameters
        ----------
        trial_budget: int
            The number of trials to sample before ending.
        max_steps : int, optional
            The maximum number of steps to execute, by default 1.
        space : Dict[str, Options], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        super().__init__(max_steps, space=space)
        self.trial_budget = trial_budget
        self.seed = seed

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Mapping from option name to values.

        """
        super().initialize(space)

        searcher = RandomSearcher(space, seed=self.seed)
        self.scheduler = BlackBoxScheduler(
            searcher=searcher,
            max_steps=self.max_steps,
            trial_budget=self.trial_budget
        )


class BayesOptGP(BaseAlgorithm):

    def __init__(self,
                 trial_budget: int,
                 max_steps: int = 1,
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
            [description]
        max_steps : int, optional
            [description], by default 1
        min_configs_in_model : int, optional
            [description], by default 1
        aq_func : str, optional
            [description], by default "ei"
        kappa : float, optional
            [description], by default 2.5
        xi : float, optional
            [description], by default 0.0
        seed : int, optional
            [description], by default None
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.e

        """
        super().__init__(max_steps, space=space)

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
            Mapping from option name to values.

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
            trial_budget=self.trial_budget,
            max_steps=self.max_steps,
        )


class BayesOptKDE(BaseAlgorithm):

    def __init__(self,
                 trial_budget,
                 max_steps: int = 1,
                 min_configs_per_model: Optional[int] = None,
                 top_n_frac: float = 0.15,
                 num_samples: int = 64,
                 random_fraction: float = 1 / 3,
                 bandwidth_factor: float = 3,
                 min_bandwidth: float = 1e-3,
                 space: Optional[Dict[str, Distribution]] = None):
        """The BayesOpt algorithm with KDE.

        Learns how to search over the space with KDE.

        Parameters
        ----------
        trial_budget : [type]
            [description]
        max_steps : int, optional
            [description], by default 1
        min_configs_per_model : Optional[int], optional
            [description], by default None
        top_n_frac : float, optional
            [description], by default 0.15
        num_samples : int, optional
            [description], by default 64
        random_fraction : float, optional
            [description], by default 1/3
        bandwidth_factor : float, optional
            [description], by default 3
        min_bandwidth : float, optional
            [description], by default 1e-3
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        super().__init__(max_steps, space=space)

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
            Mapping from option name to values.

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
            trial_budget=self.trial_budget,
            max_steps=self.max_steps,
        )


class Hyperband(BaseAlgorithm):

    def __init__(self,
                 step_budget: int,
                 drop_rate: int = 3,
                 max_steps: int = 1,
                 seed: Optional[int] = None,
                 space: Optional[Dict[str, Distribution]] = None):
        """The Hyperband algorithm.

        Parameters
        ----------]
        max_steps : int, optional
            [description], by default 1
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.e

        """
        super().__init__(max_steps, space=space)
        self.step_budget = step_budget
        self.seed = seed
        self.drop_rate = drop_rate

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        space: Dict[str, Distribution]
            Mapping from option name to values.

        """
        super().initialize(space)

        searcher = RandomSearcher(space, seed=self.seed)
        self.scheduler = HyperBandScheduler(
            searcher=searcher,
            step_budget=self.step_budget,
            drop_rate=self.drop_rate,
            max_steps=self.max_steps,
        )


class BOHB(BaseAlgorithm):

    def __init__(self,
                 step_budget: int,
                 drop_rate: int = 3,
                 max_steps: int = 1,
                 min_steps: int = 1,
                 top_n_frac: float = 0.15,
                 num_samples: int = 64,
                 random_fraction: float = 1 / 3,
                 bandwidth_factor: int = 3,
                 min_bandwidth: float = 1e-3,
                 min_configs_per_model: Optional[int] = None,
                 seed: Optional[int] = None,
                 space: Optional[Dict[str, Distribution]] = None):
        """The BOHB algorithm.

        Uses bayesopt for selecting new trials, and hyperband to
        schedule the trials given a budget.

        See: https://arxiv.org/abs/1807.01774.

        Parameters
        ----------
        max_steps : int, optional
            [description], by default 1
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        super().__init__(max_steps, space=space)

        self.seed = seed
        self.step_budget = step_budget
        self.drop_rate = drop_rate
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
            Mapping from option name to values.

        """
        super().initialize(space)

        searcher = MultiFidSearcher(
            space=space,
            searcher_class=BayesOptKDESearcher,
            search_kwargs={
                'min_configs_per_model': self.min_configs_per_model,
                'top_n_frac': self.top_n_frac,
                'num_samples': self.num_samples,
                'random_fraction': self.random_fraction,
                'bandwidth_factor': self.bandwidth_factor,
                'min_bandwidth': self.min_bandwidth
            }
        )
        self.scheduler = HyperBandScheduler(
            searcher=searcher,
            step_budget=self.step_budget,
            drop_rate=self.drop_rate,
            max_steps=self.max_steps,
            min_steps=self.min_steps
        )
