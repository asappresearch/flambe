import torch

from flambe.model import LogisticRegression
from torch import Tensor
from numpy import isclose

NUMERIC_PRECISION = 1e-2


def test_number_of_parameters():
    regression = LogisticRegression(2)
    assert regression.num_parameters() == 3

    regression = LogisticRegression(10)
    assert regression.num_parameters() == 11


def test_forward_pass_with_target_param():
    regression = LogisticRegression(2)
    forward = regression(Tensor([[1, 4]]), target=Tensor([[5, 6]]))
    transformed_target = forward[1]
    assert transformed_target.dtype == torch.float32
    assert isclose(transformed_target.numpy(), [[5., 6.]], rtol=NUMERIC_PRECISION).all()


def test_forward_pass_is_sigmoid():
    regression = LogisticRegression(2)

    set_parameters(regression, Tensor([[5, 6]]), Tensor([[3]]))

    assert isclose(regression(Tensor([[1, 4]])).item(),   [[1.]], rtol=NUMERIC_PRECISION)
    assert isclose(regression(Tensor([[0, 0]])).item(), [[0.95]], rtol=NUMERIC_PRECISION)
    assert isclose(regression(Tensor([[1.5, 2.5]])).item(), [[1.]], rtol=NUMERIC_PRECISION)
    assert isclose(regression(Tensor([[-1, 0]])).item(), [[0.12]], rtol=NUMERIC_PRECISION)
    assert isclose(regression(Tensor([[-0.5, 0]])).item(), [[0.62]], rtol=NUMERIC_PRECISION)


def set_parameters(model, weight, bias):
    """
    This depends and has knowledge of the inner structure of objects. According
    to https://github.com/pytorch/pytorch/issues/565 there's no plan of adding
    a feature to inject weights/biases dependencies.
    Probably we can iterate this idea to patch this.
    """
    for name, param in model.named_parameters():
        if "bias" in name:
            param.data = bias
        if "weight" in name:
            param.data = weight
