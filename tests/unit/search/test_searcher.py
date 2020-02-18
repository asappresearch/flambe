import pytest
import numpy as np

from flambe.search.searcher import GridSearcher, RandomSearcher
from flambe.search.searcher import BayesOptGPSearcher, BayesOptKDESearcher
from flambe.search.distribution import Choice, QUniform, Beta, Uniform


def test_grid():
    bad_space = {'var1': Choice(['red', 'blue', 'green']),
                 'var2': QUniform(-6, -2, 5, transform='pow10'),
                 'var3': Beta(2, 3)}
    with pytest.raises(ValueError):
        searcher = GridSearcher(space=bad_space)

    good_space = {'var1': Choice(['red', 'blue', 'green']),
                  'var2': QUniform(-6, -2, 5, transform='pow10'),
                  'var3': Choice([1, 2, 3])}
    searcher = GridSearcher(space=good_space)
    for _ in range(45):
        params = searcher._propose_new_params_in_model_space()
        assert len(params) == 3
        assert 'var1' in params.keys()
        assert 'var2' in params.keys()
        assert 'var3' in params.keys()
    params = searcher._propose_new_params_in_model_space()
    assert params is None

    d = {'var1': ['red', 'blue'], 'var2': [-6, -5]}
    lst = [{'var1': 'red', 'var2': -6},
           {'var1': 'red', 'var2': -5},
           {'var1': 'blue', 'var2': -6},
           {'var1': 'blue', 'var2': -5}]
    searcher._dict_to_cartesian_list(d) == lst


def test_random():
    space = {'var1': QUniform(-6, -2, 5, transform='pow10'),
             'var2': Choice([1, 2, 3]),
             'var3': Beta(4, 5),
             'var4': Choice(['red', 'blue', 'green'])}
    searcher = RandomSearcher(space=space, seed=1234)
    params = searcher._propose_new_params_in_model_space()
    assert len(params) == 4
    assert 'var1' in params.keys()
    assert 'var2' in params.keys()
    assert 'var3' in params.keys()
    assert 'var4' in params.keys()

    searcher = RandomSearcher(space=space, seed=1234)
    params2 = searcher._propose_new_params_in_model_space()
    assert params == params2


def test_bayesian_gp():
    bad_space = {'var1': Choice([1, 2, 3]),
                 'var2': QUniform(-6, -2, 5, transform='pow10'),
                 'var3': Beta(2, 3)}
    with pytest.raises(ValueError):
        searcher = BayesOptGPSearcher(space=bad_space)

    good_space = {'var1': Beta(2, 3),
                  'var2': QUniform(-6, -2, 5, transform='pow10'),
                  'var3': QUniform(1, 3, 3)}
    searcher = BayesOptGPSearcher(space=good_space)
    name, params = searcher.propose_new_params()
    assert len(params) == 3
    assert 'var1' in params.keys()
    assert 'var2' in params.keys()
    assert 'var3' in params.keys()

    results = {name: 3}
    searcher.register_results(results)

    searcher.propose_new_params()
    assert len(searcher.results) == 1
    assert len(searcher.params) == 2

    with pytest.raises(ValueError):
        searcher.propose_new_params()


def test_bayesian_kde():
    bad_space = {'var1': Choice([1, 2, 3]),
                 'var2': QUniform(-6, -2, 5, transform='pow10'),
                 'var3': Beta(2, 3)}
    with pytest.raises(ValueError):
        searcher = BayesOptKDESearcher(space=bad_space)

    good_space = {'var1': Beta(2, 3),
                  'var2': Uniform(0, 1, transform='exp'),
                  'var3': Choice([1, 2, 3])}
    searcher = BayesOptKDESearcher(space=good_space)

    results = {}
    n_finished = 0
    for _ in range(20):
        name, params = searcher.propose_new_params()
        results[name] = np.random.uniform()

        # Check that searcher can handle registering at any time
        if np.random.uniform() < 0.5:
            searcher.register_results(results)
            n_finished += len(results)
            results = {}
        # Check if searcher has as many results as expected
        assert len(searcher.results) == n_finished
