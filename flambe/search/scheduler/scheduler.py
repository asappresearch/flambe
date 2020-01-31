from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from flambe.search.trial import Trial
from flambe.search.searcher.searcher import Searcher


class Scheduler(ABC):
    """Base scheduler class.

    Schedulers only observe the results of trials and may decide
    to temporarily freeze or permanently end trials based on these
    results. They report the results back to the searcher for future
    hyperparameter configuration proposals.

    """

    def __init__(self, searcher: Searcher, max_steps: int = 1):
        """Initialize a scheduler.

        Parameters
        ----------
        searcher: Searcher
            The searcher object to use to explore the search space.
        max_steps: int
            Maximum number of steps that will be logged.

        """
        self.trials: Dict[str, Trial] = {}
        self.max_steps = max_steps
        self.searcher = searcher
        self.done = False

    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        ----------
        bool
            ``True`` if the algorithm has terminated.

        """
        return self.done

    def _create_trial(self) -> Optional[Tuple[str, Trial]]:
        """Create a new trial.

        Returns
        ----------
        str, optional
            The name of the new trial.
        Trial, optional
            The new trial object.

        """
        # Create trial
        new = self.searcher.propose_new_params()
        if new is None:
            return None
        else:
            name, params = new
            new_trial = Trial(params)
            return name, new_trial

    @abstractmethod
    def release_trials(self, n: int, trials: Dict[str, Trial]) -> Dict[str, Trial]:
        """Release trials with hyperparameter configurations.

        New trials may be added or existing trials may be unfrozen.

        Parameters
        ----------
        n: int
            The number of new trials to add.
        trials: Dict[str, Trial]
            A dictionary mapping trial name to corresponding Trial
            objects.

        Returns
        ----------
        Dict[str, Trial]
            A dictionary mapping trial name to corresponding Trial
            objects, with newly released trials.

        """
        pass

    @abstractmethod
    def update_trials(self, trials: Dict[str, Trial]) -> Dict[str, Trial]:
        """Update the algorithm with trial results.

        Parameters
        ----------
        trials: Dict[str, Trial]
            A dictionary mapping trial id to corresponding Trial
            objects.

        Returns
        ----------
        Dict[str, Trial]
            A dictionary mapping trial id to corresponding Trial
            objects.

        """
        pass
