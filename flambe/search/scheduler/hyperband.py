import numpy as np
from typing import List, Dict, Tuple, Optional, Sequence
from copy import deepcopy

from flambe.search.searcher.searcher import Searcher, ModelBasedSearcher
from flambe.search.scheduler.scheduler import Scheduler
from flambe.search.trial import Trial


class HyperBandScheduler(Scheduler):
    """The HyperBand scheduling algorithm."""

    def __init__(self,
                 searcher: Searcher,
                 step_budget: int,
                 max_steps: int,
                 min_steps: int = 1,
                 drop_rate: float = 3):
        """Initializes the HyperBand scheduler.

        Parameters
        ----------
        searcher: Searcher
            The searcher object to use to explore the search space.
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
        super().__init__(searcher, max_steps)

        if isinstance(searcher, ModelBasedSearcher):
            self.searchers: Dict[int, Searcher] = dict()
            self.searchers[min_steps] = searcher
            self.max_fid = min_steps
            self.base_searcher = deepcopy(searcher)

        self.max_steps_by_trial: Dict[str, int] = dict()

        self.min_steps = min_steps
        self.drop_rate = drop_rate

        # Initialize HyperBand params
        self.max_drops = int(np.log(max_steps / min_steps) / np.log(drop_rate))
        n_drops_grid = np.arange(self.max_drops + 1)
        self.n_trials_grid = np.ceil((self.max_drops + 1) / (n_drops_grid + 1) * (drop_rate**n_drops_grid)).astype('int')
        self.n_res_grid = max_steps / (drop_rate**n_drops_grid)

        self.step_budget = step_budget
        self.n_bracket_runs = self.step_budget // ((self.max_drops + 1) * self.max_steps)

        # Keep track of each bracket
        self.brackets = []
        for br in range(self.n_bracket_runs):
            s = self.max_drops - br % (self.max_drops + 1)
            self.brackets.append(Bracket(self.n_trials_grid[s], s))

        self.bracket_ids: Dict[str, int] = {}
        self.done = False

    def is_done(self) -> bool:
        """Whether the algorithm has finished producing trials.

        Returns
        ----------
        bool
            ``True`` if the algorithm has terminated.

        """
        return self.done or all(bracket.has_finished for bracket in self.brackets)

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
        Dict[st, Trial]
            A dictionary mapping trial name to corresponding Trial
            objects, with newly released trials.

        """
        for trial_id, trial in trials.items():
            if trial.has_result():
                trial.set_resume()
                n -= 1

        for br, bracket in enumerate(self.brackets):
            s = bracket.max_halvings

            while n > 0 and (not bracket.has_finished) and bracket.has_pending:
                trial_id = bracket.get_pending()
                n_res_init = self.n_res_grid[s]

                if trial_id is None:
                    # Create new trial
                    new = self._create_trial()
                    if new is None:
                        self.done = True
                    else:
                        trial_id, trial = new
                        trials[trial_id] = trial
                        self.max_steps_by_trial[trial_id] = int(n_res_init)
                else:
                    # Resume previously paused trial
                    trial = trials[trial_id]
                    # TODO
                    n_res_total = n_res_init * (self.drop_rate ** bracket.n_halvings)
                    self.max_steps_by_trial[trial_id] = int(n_res_total)
                    trial.set_resume()

                self.bracket_ids[trial_id] = br
                n -= 1

        return trials

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
        trials_with_result = {trial_id: trial for trial_id,
                              trial in trials.items() if trial.has_result()}

        trials_with_max_steps: Dict[str, Trial] = dict()
        for trial_id, trial in trials_with_result.items():
            n_metrics = len(trial.metrics)
            assert n_metrics <= self.max_steps_by_trial[trial_id]
            if n_metrics == self.max_steps_by_trial[trial_id]:
                trials_with_max_steps[trial_id] = trial
                trial.set_paused()

        if isinstance(self.searcher, ModelBasedSearcher):
            # Register results with appropriate model-based searcher
            results_groups: Dict[int, Dict[str: float]] = dict()
            for trial_id, trial in trials_with_max_steps.items():
                fid = self.max_steps_by_trial[trial_id]
                result = trial.best_metric
                if fid in results_groups:
                    results_groups[fid][trial_id] = result
                else:
                    results_groups[fid] = {trial_id: result}

            for fid, results in results_groups.items():
                # Create new searcher if necessary
                if fid not in self.searchers:
                    self.searchers[fid] = deepcopy(self.base_searcher)
                searcher = self.searchers[fid]

                # Make sure searcher has parameter configuration saved
                for trial_id in results.keys():
                    if trial_id not in searcher.params:
                        params = trials[trial_id].params
                        searcher.params[trial_id] = params

                # Update searcher with results
                searcher.register_results(results)

                # Update best searcher
                if fid > self.max_fid and searcher.has_enough_configs_for_model:
                    self.max_fid = fid
                    self.searcher = searcher
        else:
            # Register results with searcher
            results_dict = {tid: trial.best_metric for tid, trial in trials_with_max_steps.items()}
            self.searcher.register_results(results_dict)

        # Update brackets with results
        for trial_id, trial in trials_with_max_steps.items():
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
    """Internal HyperBand data structure."""

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
        self.pending: Sequence[Optional[str]] = [None] * n_trials
        self.finished: List[Tuple[str, float]] = []

    def get_pending(self):
        """Get a pending trial id.

        Returns
        ----------
        int
            The trial id.

        """
        return self.pending.pop()

    def record_finished(self, trial: str, result: float):
        """Record the result of a finished trial.

        Parameters
        ----------
        trial: str
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
        """Check if the bracket is ready for a drop episode (also
        called 'halving').

        Returns
        -------
        bool
            Denotes whether or not the bracket is ready to drop.

        """
        return (self.n_halvings < self.max_halvings) & (self.n_trials == len(self.finished))

    @property
    def has_finished(self) -> bool:
        """Check if the bracket has finished processing all its trials.

        Returns
        -------
        bool
            Denotes whether or not the bracket is finished.

        """
        return (self.n_halvings == self.max_halvings) & (self.n_trials == len(self.finished))

    def execute_halving(self, n_active: int) -> List[str]:
        """Run a drop episode, which only keeps (approximately) the top
        1/drop_rate trials.  All other trials are ended.

        Parameters
        ----------
        n_active: int
            The number of trials to keep active.

        Returns
        ----------
        List[str]
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
