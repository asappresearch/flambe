from typing import Dict

from flambe.search.searcher.searcher import Searcher
from flambe.search.scheduler.scheduler import Scheduler
from flambe.search.trial import Trial


class BlackBoxScheduler(Scheduler):
    """A BlackBoxScheduler.

    Simply observes the final result of trials and reports them
    back to the searcher. Does not end trials until completion.

    """

    def __init__(self,
                 searcher: Searcher,
                 trial_budget: int) -> None:
        """Initialize the BlackBoxScheduler.

        Parameters
        ----------
        searcher: Searcher
            The searcher object to use to explore the search space.
        trial_budget: int
            The maximum number of trials to create.
        max_steps: int
            The maximum number of steps to give each trial.

        """
        super().__init__(searcher)
        self.trial_budget = trial_budget
        self.num_terminated = 0

    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        -------
        bool
            ``True`` if the algorithm has terminated.

        """
        return self.done or (self.num_terminated >= self.trial_budget)

    def release_trials(self, n: int, trials: Dict[str, Trial]) -> Dict[str, Trial]:
        """Update the list of trials with new ones.

        Always creates `n` new trials if the budget allows it.

        Parameters
        ----------
        n: int
            The number of new trials to add.
        trials: Dict[str, Trial]
            A dictionary mapping trial name to corresponding Trial.

        Returns
        ----------
        Dict[str, Trial]
            A dictionary mapping trial name to corresponding Trial
            objects, including newly released trials.

        """
        # Process running trials
        for trial_id, trial in trials.items():
            if trial.has_result():
                trial.set_resume()
                n -= 1

        # Create new trials
        while n > 0 and len(trials) < self.trial_budget:
            new = self._create_trial()
            if new is None:
                self.done = True
                break
            else:
                trial_id, trial = new
                trials[trial_id] = trial
                n -= 1

        return trials

    def update_trials(self, trials: Dict[str, Trial]) -> Dict[str, Trial]:
        """Update the algorithm with trial results.

        Simply passes the trial results back to the searcher.

        Parameters
        ----------
        trials: Dict[str, Trial]
            A dictionary mapping trial id to corresponding Trial.

        Returns
        ----------
        Dict[str, Trial]
            A dictionary mapping trial id to corresponding Trial.

        """
        results: Dict[str, float] = dict()
        num_terminated = 0

        # Process results and terminations
        for trial_id, trial in trials.items():
            if trial.has_result():
                results[trial_id] = trial.best_metric
                trial.set_resume()
            elif trial.is_error() or trial.is_terminated():
                num_terminated += 1

        self.num_terminated = num_terminated
        self.searcher.register_results(results)  # type: ignore
        return trials
