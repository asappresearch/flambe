import numpy as np

from flambe.search.searcher.searcher import Space
from flambe.search.distribution import Choice, Beta, QUniform


def test_space():

    distributions = {'var1': Choice(['red', 'blue', 'green']),
                     'var2': Beta(2, 3, transform='exp'),
                     'var3': QUniform(-6, -2, 5,
                                      transform=lambda x: x**2)}
    space = Space(distributions)

    raw_dist_samp = space.sample()
    assert sorted(raw_dist_samp.keys()) == sorted(distributions.keys())

    for key, val in raw_dist_samp.items():
        low, high = space.var_bounds[key]
        if key == 'var1':
            assert np.isnan(low) and np.isnan(high)
        else:
            assert low <= val and high >= val

    transformed_samp = {'var1': raw_dist_samp['var1'],
                        'var2': np.exp(raw_dist_samp['var2']),
                        'var3': raw_dist_samp['var3'] ** 2}
    assert transformed_samp == space.apply_transform(raw_dist_samp)

    raw_dist_samp = {'var1': 'red', 'var2': 0.7, 'var3': -3.8}
    rounded_samp = {'var1': 'red', 'var2': 0.7, 'var3': -4.0}
    assert space.round_to_space(raw_dist_samp) == rounded_samp

    normalized_samp = {'var1': 0, 'var2': 0.7, 'var3': 0.5}
    assert space.normalize_to_space(rounded_samp) == normalized_samp

    identity = lambda x: space.unnormalize(space.normalize_to_space(x))  # noqa: E731
    assert identity(rounded_samp) == rounded_samp
