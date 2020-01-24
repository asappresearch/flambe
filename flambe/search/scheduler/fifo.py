from typing import List, Dict, Any, Optional

from flambe.search.scheduler.scheduler import Scheduler
from flambe.search.trial import Trial


class BlackBoxScheduler(Scheduler):
    """
    A BlackBoxScheduler simply observes the final result of trials and
    reports them back to the searcher.
    """

    def __init__(self,
                 trial_budget: int,
                 max_steps: int = 1):
        """Initialize the BlackBoxScheduler.

        Parameters
        ----------
        trial_budget: int
            The maximum number of trials to create.
        max_steps: int
            The maximum number of steps to give each trial.
        """
        super().__init__(max_steps)
        self.trial_budget = trial_budget
        self.num_released = 0
        self.done = False

    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        ----------
        bool
            ``True`` if the algorithm has terminated.

        """
        return self.done or (self.num_released >= self.trial_budget)

    def release_trials(self,
                       n: int,
                       trials: Dict[int, Trial]) -> Dict[int, Trial]:
        """
        Release trials with hyperparameter configurations
        to the trial dictionary.  Always creates `n` new trials if the
        budget allows.

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
        while len(trials) < self.trial_budget and n > 0:
            trial_id, trial = self._create_trial(self.max_steps)
            if trial is None:
                self.done = True
                break
            else:
                trials[trial_id] = trial
                n -= 1
                self.num_released += 1
        return trials

    def update_trials(self,
                      trials: Dict[int, Trial]) -> Dict[int, Trial]:
        """Update the algorithm with trial results.  Simply passes the
        trial results back to the searcher.

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
        trials_with_result = {trial_id: trial for trial_id,
                              trial in trials.items() if trial.has_result()}
        results_dict = {t_id: trial.best_metric for t_id, trial in trials_with_result.items()}
        self._register_results(results_dict)
        for trial in trials_with_result.values():
            trial.set_resume()
        return trials
