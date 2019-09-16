import pytest
import torch
import torch.testing

from torch import Tensor
from flambe.nn import pooling


TENSOR = Tensor(
    [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ],
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
    ]
)


MASK = Tensor(
    [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
    ]
)


@pytest.mark.parametrize("size", [(10, 20, 30),
                                  (10, 10, 10),
                                  (30, 20, 10),
                                  (1, 20, 30),
                                  (20, 1, 30),
                                  (1, 1, 1),
                                  (0, 10, 1)])
@pytest.mark.parametrize("pooling_cls", [pooling.LastPooling, pooling.FirstPooling,
                                         pooling.AvgPooling, pooling.SumPooling])
def test_last_pooling(pooling_cls, size):
    _in = torch.rand(*size)
    p = pooling_cls()
    out = p(_in)

    assert len(out.size()) == len(_in.size()) - 1
    assert out.size() == torch.Size([_in.size(0), _in.size(-1)])


@pytest.mark.parametrize("pooling_cls", [pooling.LastPooling, pooling.FirstPooling,
                                         pooling.AvgPooling, pooling.SumPooling])
def test_invalid_pooling(pooling_cls):
    """Test that last pooling fails where there is no sequence"""
    _in = torch.rand(10, 0, 20)
    p = pooling_cls()
    with pytest.raises(ValueError):
        _ = p(_in)


@pytest.mark.parametrize("mask", [[0.5, 0, 1],
                                  [2, 3, 1],
                                  [0, 0, -1]])
@pytest.mark.parametrize("pooling_cls", [pooling.LastPooling, pooling.FirstPooling,
                                         pooling.AvgPooling, pooling.SumPooling])
def test_invalid_mask_1(pooling_cls, mask):
    t = torch.rand(1, 2, 4)
    mask = Tensor([mask])

    p = pooling_cls()
    with pytest.raises(ValueError):
        _ = p(t, padding_mask=mask)


@pytest.mark.parametrize("mask", [torch.ones(10, 30), torch.ones(20, 30)])
@pytest.mark.parametrize("pooling_cls", [pooling.LastPooling, pooling.FirstPooling,
                                         pooling.AvgPooling, pooling.SumPooling])
def test_invalid_mask_2(pooling_cls, mask):
    """Test that last pooling fails where there is no sequence"""
    _in = torch.rand(10, 20, 30)
    p = pooling_cls()
    with pytest.raises(ValueError):
        _ = p(_in, padding_mask=mask)


def test_last_pooling_with_mask():
    lp = pooling.LastPooling()
    out = lp(TENSOR, padding_mask=MASK)

    assert out.size() == torch.Size([2, 4])

    expected = Tensor(
        [
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ]
    )
    torch.testing.assert_allclose(out, expected)


def test_first_pooling_with_mask():
    lp = pooling.FirstPooling()
    out = lp(TENSOR, padding_mask=MASK)

    assert out.size() == torch.Size([2, 4])

    expected = Tensor(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        ]
    )
    torch.testing.assert_allclose(out, expected)


def test_sum_pooling_with_mask():
    lp = pooling.SumPooling()
    out = lp(TENSOR, padding_mask=MASK)

    assert out.size() == torch.Size([2, 4])

    expected = Tensor(
        [
            [6, 8, 10, 12],
            [15, 18, 21, 24]
        ]
    )
    torch.testing.assert_allclose(out, expected)


def test_avg_pooling_with_mask():
    lp = pooling.AvgPooling()
    out = lp(TENSOR, padding_mask=MASK)

    assert out.size() == torch.Size([2, 4])

    expected = Tensor(
        [
            [3, 4, 5, 6],
            [5, 6, 7, 8]
        ]
    )
    torch.testing.assert_allclose(out, expected)
