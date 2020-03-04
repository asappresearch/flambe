import pytest
import torch
import torch.testing


from torch import Tensor, allclose
from flambe.nn import pooling

from flambe.nn import AvgPooling, SumPooling, StructuredSelfAttentivePooling, \
    GeneralizedPooling

from pytest import approx


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


def test_avg_pooling():
    layer = AvgPooling()
    average = layer.forward(build_tensor())
    assert allclose(average[0], torch.tensor([10 / 4, 14 / 4, 18 / 4], dtype=torch.float32))
    assert allclose(average[1], torch.tensor([0, 0, 0], dtype=torch.float32))


def test_sum_pooling():
    layer = SumPooling()
    summation = layer.forward(build_tensor())
    assert allclose(summation[0], torch.tensor([10, 14, 18], dtype=torch.float32))
    assert allclose(summation[1], torch.tensor([0, 0, 0], dtype=torch.float32))


def test_structured_self_attentive_pooling_shapes():
    dim = 300
    layer = StructuredSelfAttentivePooling(input_size=dim)
    input = torch.randn(100, 50, dim)
    output = layer(input)
    assert list(output.shape) == [100, dim]


def test_structured_self_attentive_pooling_zeroes():
    dim = 300
    layer = StructuredSelfAttentivePooling(input_size=dim)
    input = torch.zeros(100, 50, dim)
    output = layer(input)
    assert list(output.shape) == [100, dim]
    assert output.min().item() == output.max().item() == approx(0.)


def test_structured_self_attentive_pooling_ones():
    dim = 300
    layer = StructuredSelfAttentivePooling(input_size=dim)
    input = torch.ones(100, 50, dim)
    output = layer(input)
    assert list(output.shape) == [100, dim]
    assert output.min().item() == output.max().item() == approx(1.)


def test_vector_based_generalized_pooling_shapes():
    dim = 300
    layer = GeneralizedPooling(input_size=dim)
    input = torch.randn(100, 50, dim)
    output = layer(input)
    assert list(output.shape) == [100, dim]


def build_tensor():
    batch_size = 2
    items = 4
    embedding_size = 3
    data = torch.zeros([batch_size, items, embedding_size], dtype=torch.float32)
    data[0][0] = torch.tensor([1, 2, 3])
    data[0][1] = torch.tensor([4, 5, 6])
    data[0][2] = torch.tensor([1, 2, 3])
    data[0][3] = torch.tensor([4, 5, 6])
    return data


if __name__ == '__main__':
    test_vector_based_generalized_pooling_shapes()
