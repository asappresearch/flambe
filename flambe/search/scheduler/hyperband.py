import numpy as np
from typing import List, Dict, Any, Optional, Callable

from flambe.search.scheduler.scheduler import Scheduler
from flambe.search.trial import Trial


class HyperBandScheduler(Scheduler):
    """The HyperBand scheduling algorithm.
    """
    def __init__(self,
                 step_budget: int,
                 max_steps: int,
                 min_steps: int = 1,
                 drop_rate: float = 3):
        """Initializes the HyperBand scheduler.

        Parameters
        ----------
        step_budget: int
            The maximum number of steps to allocate across all trials.
        max_steps: int
            The maximum number of steps that can be allocated to a
            single trial.
        min_steps: int
            The minimum number of steps that can be allocated to a
            single trial.
        drop_rate: float
            The rate at which trials are dropped between rounds of the
            HyperBand algorithm.  A higher drop rate means that the
            algorithm will be more exploitative than exploratory.
        """
        super().__init__(max_steps)
        self.min_steps = min_steps
        self.drop_rate = drop_rate

        # Initialize HyperBand params
        self.max_drops = int(np.log(max_steps / min_steps) / np.log(drop_rate))
        n_drops_grid = np.arange(self.max_drops + 1)
        self.n_trials_grid = np.ceil((self.max_drops + 1) / (n_drops_grid + 1) *
                                     (drop_rate**n_drops_grid)).astype('int')
        self.n_res_grid = max_steps / (drop_rate**n_drops_grid)

        self.step_budget = step_budget
        self.n_bracket_runs = self.step_budget // ((self.max_drops + 1) * self.max_steps)

        # Keep track of each bracket
        self.brackets = []
        for br in range(self.n_bracket_runs):
            s = self.max_drops - br % (self.max_drops + 1)
            self.brackets.append(Bracket(self.n_trials_grid[s], s))

        self.bracket_ids = {}

    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        ----------
        bool
            ``True`` if the algorithm has terminated.

        """
        return all(bracket.has_finished for bracket in self.brackets)

    def release_trials(self,
                       n: int,
                       trials: Dict[int, Trial]) -> Dict[int, Trial]:
        """
        Release trials with hyperparameter configurations
        to the trial dictionary.  May add new trials or unfreeze
        old trials.

        Parameters
        ----------
        n: int
            The number of trials to release.
        trials: Dict[int, Trial]
            A dictionary mapping trial id to corresponding Trial
            objects.

        Returns
        ----------
        Dict[int, Trial]
            A dictionary mapping trial id to corresponding Trial
            objects, with newly released trials.
        """
        for br, bracket in enumerate(self.brackets):
            s = bracket.max_halvings

            while n > 0 and (not bracket.has_finished) and bracket.has_pending:
                trial_id = bracket.get_pending()
                n_res_init = self.n_res_grid[s]

                if trial_id is None:
                    # Create new trial
                    trial_id, trial = self._create_trial(int(n_res_init))
                    trials[trial_id] = trial
                else:
                    # Resume previously paused trial
                    trial = trials[trial_id]
                    # TODO
                    # n_res_curr = len(trial.metrics)
                    # n_res_total = n_res_init *
                    # (self.drop_rate**bracket.n_halvings)
                    # trial.append_steps(int(n_res_total - n_res_curr))
                    trial.set_resume()

                self.bracket_ids[trial_id] = br
                n -= 1

        return trials

    def update_trials(self, trials):
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

        # Send results back to searcher
        trials_with_result = {trial_id: trial for trial_id,
                              trial in trials.items() if trial.has_result()}
        results_dict = {t_id: trial.best_metric for t_id, trial in trials_with_result.items()}
        fids_dict = {t_id: len(trial.metrics) for t_id, trial in trials_with_result.items()}
        self._register_results(results_dict, fids_dict=fids_dict)

        # Update brackets with results
        for trial_id, trial in trials_with_result.items():
            br = self.bracket_ids[trial_id]
            bracket = self.brackets[br]
            s = bracket.max_halvings
            bracket.record_finished(trial_id, trial.get_metric(-1))
            trial.set_paused()

            if bracket.is_ready_for_halving:
                n_trials_init = self.n_trials_grid[s]
                i = bracket.n_halvings + 1
                n_active = int(n_trials_init / (self.drop_rate**i))
                non_active_trial_ids = bracket.execute_halving(n_active)
                for trial_id in non_active_trial_ids:
                    trials[trial_id].set_terminated()

            elif bracket.has_finished:
                trial_ids = [x[0] for x in bracket.finished]
                for trial_id in trial_ids:
                    trials[trial_id].set_terminated()
        return trials


class Bracket:
    """Internal HyperBand data structure.
    """

    def __init__(self, n_trials: int, max_halvings: int):
        """Initialize the bracket.

        Parameters
        ----------
        n_trials: int
            The number of trials that the bracket holds.
        max_halvings: int
            The number of drop episodes that the bracket will execute.
        """
        self.n_trials = n_trials
        self.n_halvings = 0
        self.max_halvings = max_halvings
        self.pending = [None] * n_trials
        self.finished = []

    def get_pending(self):
        """Get a pending trial id.

        Returns
        ----------
        int
            The trial id.
        """
        return self.pending.pop()

    def record_finished(self, trial: int, result: float):
        """Record the result of a finished trial.

        Parameters
        ----------
        trial: int
            The trial id.
        result:
            The corresponding result of the trial.
        """
        self.finished.append((trial, result))

    @property
    def has_pending(self) -> bool:
        """Check if the bracket has a pending trial.

        Returns
        ----------
        bool
            Denotes whether or not the bracket has a pending trial.
        """
        return len(self.pending) > 0

    @property
    def is_ready_for_halving(self) -> bool:
        """
        Check if the bracket is ready for a drop episode (also
        called 'halving').

        Returns
        ----------
        bool
            Denotes whether or not the bracket is ready to drop.
        """
        return (self.n_halvings < self.max_halvings) & (self.n_trials == len(self.finished))

    @property
    def has_finished(self) -> bool:
        """
        Check if the bracket has finished processing all of its trials.

        Returns
        ----------
        bool
            Denotes whether or not the bracket is finished.
        """
        return (self.n_halvings == self.max_halvings) & (self.n_trials == len(self.finished))

    def execute_halving(self, n_active: int) -> List[int]:
        """
        Run a drop episode, which only keeps (approximately) the top
        1/drop_rate trials.  All other trials are ended.

        Parameters
        ----------
        n_active: int
            The number of trials to keep active.

        Returns
        ----------
        List[int]
            The ids of the trials to end.
        """
        trials_sorted = sorted(self.finished, key=lambda x: x[1], reverse=True)
        trials_sorted = [x[0] for x in trials_sorted]
        active_trials = trials_sorted[:n_active]
        non_active_trials = trials_sorted[n_active:]

        self.n_trials = n_active
        self.pending = active_trials
        np.random.shuffle(self.pending)

        self.finished = []

        self.n_halvings += 1
        return non_active_trials
