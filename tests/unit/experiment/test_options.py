import pytest

from flambe.experiment import options


def test_g_tag():
    l = list(range(5))
    grid = options.GridSearchOptions.from_sequence(l)

    for i, each in enumerate(grid):
        assert each == l[i]


def test_s_tag():
    # Test k

    for k in range(5, 1):
        l = [1, 10, k]
        opts = options.SampledUniformSearchOptions.from_sequence(l)

        assert len(opts) == k


def test_s_tag_2():
    # Test int sampling
    l = [1, 10, 1, 10]

    opts = options.SampledUniformSearchOptions.from_sequence(l)

    for each in opts:
        assert isinstance(each, int)



def test_s_tag_3():
    # Test float sampling

    l = [1, 10.1, 1]

    opts = options.SampledUniformSearchOptions.from_sequence(l)

    for each in opts:
        assert isinstance(each, float)

    l = [1.1, 10.1, 1]

    opts = options.SampledUniformSearchOptions.from_sequence(l)

    for each in opts:
        assert isinstance(each, float)


    l = [1.1, 10, 1]

    opts = options.SampledUniformSearchOptions.from_sequence(l)

    for each in opts:
        assert isinstance(each, float)


def test_s_tag_4():
    # Test decimals
    d = 5
    l = [1, 10.1, 5, d]

    opts = options.SampledUniformSearchOptions.from_sequence(l)

    for each in opts:
        decimals = len(str(each)[str(each).find('.') + 1:])
        assert decimals <= d


def test_s_tag_incorrect_params():
    # Test decimals
    l = [1, 10.1, 5, 1, 1]

    with pytest.raises(ValueError):
        options.SampledUniformSearchOptions.from_sequence(l)


    l = [1]

    with pytest.raises(ValueError):
        options.SampledUniformSearchOptions.from_sequence(l)
