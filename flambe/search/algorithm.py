from abc import abstractmethod
from typing import Dict, Optional

from flambe.compile import RegisteredStatelessMap
from flambe.search.distribution import Distribution
from flambe.search.trial import Trial
from flambe.search.searcher import Searcher, GridSearcher, RandomSearcher,\
    BayesOptGPSearcher, BayesOptKDESearcher, MultiFidSearcher
from flambe.search.scheduler import Scheduler, BlackBoxScheduler, HyperBandScheduler


class Algorithm(RegisteredStatelessMap):
    """Interface for hyperparameter search algorithms."""

    @abstractmethod
    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space.

        Parameters
        ----------
        trials:
            Mapping from option name to values.

        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        ----------
        bool
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
                 searcher: Searcher,
                 scheduler: Scheduler,
                 space: Optional[Dict[str, Distribution]] = None):
        """Initialize a BaseAlgorithm.

        Parameters
        ----------
        searcher : Searcher
            The searcher object to use. See ``Searcher``.
        scheduler : Scheduler
            The scheduler object to use. See ``Scheduler``.
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        scheduler.link_searcher_fns(searcher.propose_new_params, searcher.register_results)

        self.searcher = searcher
        self.scheduler = scheduler
        self._initialized = False

        if space is not None:
            self.initialize(space)
            self._initialized = True

    def initialize(self, space: Dict[str, Distribution]):
        """Initialize the algorithm with a search space."""
        if self._initialized:
            raise ValueError("Algorithm was already initialized.")
        self.searcher.assign_space(space)

    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials."""
        return self.scheduler.is_done()

    def update(self, trials: Dict[str, Trial], maximum: int) -> Dict[str, Trial]:
        """Update the state of the search."""
        if self.searcher.space is None:
            raise ValueError('Algorithm has not been initialized with search space.')

        trials = self.scheduler.update_trials(trials)
        trials = self.scheduler.release_trials(maximum, trials)
        return trials


class GridSearch(BaseAlgorithm):

    def __init__(self,
                 max_steps: int = 1,
                 space: Optional[Dict[str, Distribution]] = None):
        """The grid search algorithm.

        Simply runs the cross product of all search options. Only
        supports Choice objects as distributions, and the probabilities
        will be ignored.

        Parameters
        ----------
        max_steps : int, optional
            The maximum number of steps to execute, by default 1.
        space : Dict[str, Distribution], optional
            A space of hyperparameters to search over. Can also be
            provided through the initialize method.

        """
        searcher = GridSearcher()
        scheduler = BlackBoxScheduler(
            trial_budget=float('inf'),
            max_steps=max_steps,
        )
        super().__init__(searcher, scheduler)


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
        searcher = RandomSearcher(seed=seed)
        scheduler = BlackBoxScheduler(
            max_steps=max_steps,
            trial_budget=trial_budget
        )
        super().__init__(searcher, scheduler, space=space)


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
        searcher = BayesOptGPSearcher(
            min_configs_in_model=min_configs_in_model,
            aq_func=aq_func,
            kappa=kappa,
            xi=xi,
            seed=seed
        )
        scheduler = BlackBoxScheduler(
            trial_budget=trial_budget,
            max_steps=max_steps,
        )
        super().__init__(searcher, scheduler, space=space)


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
        searcher = BayesOptKDESearcher(
            min_configs_per_model=min_configs_per_model,
            top_n_frac=top_n_frac,
            num_samples=num_samples,
            random_fraction=random_fraction,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth
        )
        scheduler = BlackBoxScheduler(
            trial_budget=trial_budget,
            max_steps=max_steps
        )
        super().__init__(searcher, scheduler, space=space)


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
        searcher = RandomSearcher(seed=seed)
        scheduler = HyperBandScheduler(
            step_budget=step_budget,
            drop_rate=drop_rate,
            max_steps=max_steps,
        )
        super().__init__(searcher, scheduler, space=space)


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
        searcher = MultiFidSearcher(
            BayesOptKDESearcher,
            {
                'min_configs_per_model': min_configs_per_model,
                'top_n_frac': top_n_frac,
                'num_samples': num_samples,
                'random_fraction': random_fraction,
                'bandwidth_factor': bandwidth_factor,
                'min_bandwidth': min_bandwidth
            }
        )
        scheduler = HyperBandScheduler(
            step_budget=step_budget,
            drop_rate=drop_rate,
            max_steps=max_steps,
            min_steps=min_steps
        )
        super().__init__(searcher, scheduler, space=space)
