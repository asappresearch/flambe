import numpy as np

from flambe.search.searcher import GridSearcher, RandomSearcher, BayesOptKDESearcher
from flambe.search.scheduler import BlackBoxScheduler, HyperBandScheduler
from flambe.search.distribution import Choice, QUniform, Beta


def set_random_metrics(trials):
    for t in trials.values():
        if t.is_created() or t.is_resuming():
            metric = np.random.uniform()
            t.set_metric(metric)
            t.set_has_result()


def test_blackbox():

    good_space = {'var1': Choice(['red', 'blue', 'green']),
                  'var2': QUniform(-10, 10, transform='pow10'),
                  'var3': Choice([1, 2, 3])}
    searcher = GridSearcher(space=good_space)
    scheduler = BlackBoxScheduler(searcher, 4, max_steps=2)

    trials = {}
    trials = scheduler.release_trials(2, trials)

    # First step
    assert len(trials) == 2
    for t in trials.values():
        assert t.is_created()
    assert not scheduler.is_done()

    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)

    # Second step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 2
    for t in trials.values():
        assert t.is_resuming()
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)

    # Third step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 4
    assert len([t for t in trials.values() if t.is_created()]) == 2
    assert len([t for t in trials.values() if t.is_resuming()]) == 0
    assert len([t for t in trials.values() if t.is_terminated()]) == 2
    assert not scheduler.is_done()
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)

    # Fourth step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 4
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)
    assert len([t for t in trials.values() if t.is_terminated()]) == 4
    assert scheduler.is_done()


def test_hyperband():
    space = {'var1': QUniform(-6, -2, 5, transform='pow10'),
             'var2': Choice([1, 2, 3]),
             'var3': Beta(4, 5),
             'var4': Choice(['red', 'blue', 'green'])}
    searcher = RandomSearcher(space=space, seed=1234)
    scheduler = HyperBandScheduler(searcher, step_budget=18,
                                   max_steps=3, min_steps=1,
                                   drop_rate=3)
    assert scheduler.n_bracket_runs == 3

    # First step
    trials = {}
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 2
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)
    assert not scheduler.is_done()

    # Second step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 4
    assert len([t for t in trials.values() if t.is_paused()]) == 2
    assert len([t for t in trials.values() if t.is_created()]) == 2
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)  # Halving should occur
    assert not scheduler.is_done()

    # Third step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 4
    assert len([t for t in trials.values() if t.is_terminated()]) == 2
    assert len([t for t in trials.values() if t.is_resuming()]) == 2
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)

    # Fourth step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 4
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)
    assert scheduler.brackets[0].has_finished

    # Fifth step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 6
    assert len([t for t in trials.values() if t.is_terminated()]) == 3
    assert len([t for t in trials.values() if t.is_paused()]) == 1
    assert len([t for t in trials.values() if t.is_created()]) == 2
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)

    # Sixth step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 7
    assert len([t for t in trials.values() if t.is_created()]) == 1
    assert len([t for t in trials.values() if t.is_paused()]) == 2
    assert len([t for t in trials.values() if t.is_resuming()]) == 1
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)

    # Seventh step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 8
    assert len([t for t in trials.values() if t.is_created()]) == 1
    assert len([t for t in trials.values() if t.is_paused()]) == 3
    assert len([t for t in trials.values() if t.is_resuming()]) == 1
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)  # Halving should occur
    assert scheduler.brackets[1].has_finished
    assert not scheduler.is_done()

    # Eighth step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 8
    assert len([t for t in trials.values() if t.is_terminated()]) == 7
    assert len([t for t in trials.values() if t.is_resuming()]) == 1
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)
    assert not scheduler.is_done()

    # Ninth step
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 8
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)

    assert len(trials) == 8
    assert len([t for t in trials.values() if t.is_terminated()]) == 8
    assert scheduler.is_done()


def test_bohb():
    space = {'var1': Beta(2, 3)}
    searcher = BayesOptKDESearcher(space=space)
    scheduler = HyperBandScheduler(searcher, step_budget=18,
                                   max_steps=3, min_steps=1,
                                   drop_rate=3)
    assert scheduler.n_bracket_runs == 3

    # First step
    trials = {}
    trials = scheduler.release_trials(2, trials)
    assert len(trials) == 2
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)
    assert not scheduler.is_done()
    assert len(scheduler.searchers) == 1
    assert scheduler.max_fid == 1

    # Second step
    trials = scheduler.release_trials(2, trials)
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)  # Halving should occur
    assert len(scheduler.searchers) == 1
    assert scheduler.max_fid == 1

    # Third step
    trials = scheduler.release_trials(2, trials)
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)
    assert len(scheduler.searchers) == 1
    assert scheduler.max_fid == 1

    # Fourth step
    trials = scheduler.release_trials(2, trials)
    set_random_metrics(trials)
    trials = scheduler.update_trials(trials)
    assert len(scheduler.searchers) == 2
    assert scheduler.max_fid == 1

    for _ in range(5):
        trials = scheduler.release_trials(2, trials)
        set_random_metrics(trials)
        trials = scheduler.update_trials(trials)

    assert len(scheduler.searchers) == 2
    assert scheduler.max_fid == 3
    assert scheduler.is_done()
