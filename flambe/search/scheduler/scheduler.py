from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

from flambe.search.trial import Trial


class Scheduler(ABC):
    """
    Base scheduler class.  Schedulers only observe the results of
    trials and may decide to temporarily freeze or permanently end
    trials based on these results.  They report the results back to the
    searcher for future hyperparameter configuration proposals.
    """

    def __init__(self, max_steps: int = 1):
        """
        Parameters
        ----------
        max_steps: int
            Maximum number of steps that will be logged.
        """
        self.trials = {}
        self.max_steps = max_steps

        self._propose_new_params = None
        self._register_results = None

    def link_searcher_fns(self,
                          propose_new_params_fn: Callable,
                          register_results_fn: Callable):
        """
        Link the searcher's propose_new_params_fn and
        register_results_fn functions to the scheduler.

        Parameters
        ----------
        propose_new_params_fn: Callable
            A function for proposing new hyperparameter configurations.
        register_results_fn: Callable
            A function for recording the results of hyperparameter
            configurations.
        """
        self._propose_new_params = propose_new_params_fn
        self._register_results = register_results_fn

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
    def release_trials(self,
                       n: int,
                       trials: Dict[int, Trial]) -> Dict[int, Trial]:
        """
        Release trials with hyperparameter configurations
        to the trial dictionary.  New trials may be added or existing
        trials may be unfrozen.

        Parameters
        ----------
        n: int
            The number of new trials to add.
        trials: Dict[int, Trial]
            A dictionary mapping trial id to corresponding Trial
            objects.

        Returns
        ----------
        Dict[int, Trial]
            A dictionary mapping trial id to corresponding Trial
            objects, with newly released trials.
        """
        pass

    @abstractmethod
    def update_trials(self,
                      trials: Dict[int, Trial]) -> Dict[int, Trial]:
        """Update the algorithm with trial results.

        Parameters
        ----------
        trials: Dict[int, Trial]
            A dictionary mapping trial id to corresponding Trial
            objects.

        Returns
        ----------
        Dict[int, Trial]
            A dictionary mapping trial id to corresponding Trial
            objects.
        """
        pass

    def _create_trial(self,
                      n_steps: int, **kwargs) -> Tuple[int, Trial]:
        """Create a new trial.

        Parameters
        ----------
        n_steps: int
            The number of steps to initially assign the new trial.

        Returns
        ----------
        int
            The trial id of the new trial.
        Trial
            The new trial object.
        """
        # Create trial
        trial_id, params = self._propose_new_params(**kwargs)
        if params is None:
            return None, None
        else:
            new_trial = Trial(params)
            return trial_id, new_trial
