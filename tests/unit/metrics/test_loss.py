import math
import torch

from flambe.metric import MultiLabelCrossEntropy, MultiLabelNLLLoss


def test_cross_entropy_one_hot():
    """Test cross entropy loss when one hot"""
    y_pred = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
    y_true = torch.tensor([[1, 0], [1, 0]])

    loss = MultiLabelCrossEntropy()
    assert abs(loss(y_pred, y_true).item() - 0.70429) < 1e-2


def test_nllloss_one_hot():
    """Test negative log likelihood loss when one hot"""
    y_pred = torch.tensor([[0.2, 0.8], [0.9, 0.1]])
    y_true = torch.tensor([[1, 0], [1, 0]])

    loss = MultiLabelNLLLoss()
    assert abs(loss(y_pred, y_true).item() + 0.55) < 1e-2
