# type: ignore[override]

from typing import Optional, Sequence

import torch
from torch import nn

from flambe.nn import Module


class FirstPooling(Module):
    """Get the last hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        return data[:, 0, :]


class LastPooling(Module):
    """Get the last hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        # Compute lengths
        if padding_mask is None:
            lengths = torch.tensor([data.size(1)] * data.size(0)).long()
        else:
            lengths = padding_mask.long().sum(dim=1)

        return data[torch.arange(data.size(0)).long(), lengths - 1, :]


class SumPooling(Module):
    """Get the sum of the hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        padding_mask = padding_mask or _default_padding_mask(data)

        return _sum_with_padding_mask(data, padding_mask)


class AvgPooling(Module):
    """Get the average of the hidden state of a sequence."""

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        padding_mask = padding_mask or _default_padding_mask(data)
        value_count = padding_mask.sum(dim=1).unsqueeze(1)
        data = _sum_with_padding_mask(data, padding_mask)
        return data / value_count


class SelfAttentionPooling(Module):
    """Self attention pooling."""
    def __init__(self,
                 input_dim: int,
                 attention_heads: int = 16,
                 attention_units: Optional[Sequence[int]] = None,
                 output_activation: Optional[torch.nn.Module] = None,
                 hidden_activation: Optional[torch.nn.Module] = None,
                 input_dropout: float = 0.,
                 attention_dropout: float = 0.,
                 ):
        """Initialize a self attention pooling layer

        Parameters
        ----------
        input_dim : int
            The input data dim
        attention_heads: int
            the number of attn heads
        attention_units: Iterable[int]
            the list of hidden dimensions of the MLP computing the attn
        input_dropout: float
            dropout applied to the data argument of the forward method.
        attention_dropout: float
            dropout applied to the attention output before applying it
            to the input for reduction. decouples the attn dropout
            from the input dropout
        """
        super().__init__()
        # creating dropout applied to input
        self.in_drop = nn.Dropout(input_dropout) if input_dropout > 0. else nn.Identity()
        # creating the MLP
        # creating in, hidden and out dimensions
        dimensions = [input_dim, *attention_units, attention_heads]
        layers = []
        # iterating over hidden layers
        for l in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[l], dimensions[l+1], bias=False))
            layers.append(nn.Tanh() if hidden_activation is None else hidden_activation)
        # adding output layer
        layers.append(nn.Linear(dimensions[-2], dimensions[-1], bias=False))
        # adding attention output dropout
        if attention_dropout > 0.:
            layers.append(nn.Dropout(attention_dropout))
        # instantiating the MLP
        self.mlp = nn.Sequential(*layers)
        # instantiating the ouput layer
        self.output_activation = nn.Softmax(dim=1) \
            if output_activation is None else output_activation

    def forward(self,
                data: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        mask: torch.Tensor
            The input mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        # input_tensor is 3D float tensor, batchsize x num_encs x dim
        batch_size, num_encs, dim = data.shape
        # apply input droput
        data = self.in_drop(data)
        # apply projection and reshape to
        # batchsize x num_encs x num_heads
        attention_logits = self.mlp(data.reshape(-1, dim)).reshape(batch_size, num_encs, -1)
        # apply mask. dimension stays
        # batchsize x num_encs x num_heads
        if mask is not None:
            mask = mask.unsqueeze(2).float()
            attention_logits = attention_logits * mask + (1. - mask) * -1e20
        # apply softmax. dimension stays
        # batchsize x num_encs x num_heads
        attention = self.output_activation(attention_logits)
        # attend. attention is batchsize x num_encs x num_heads.
        # data is batchsize x num_encs x dim
        # resulting dim is batchsize x num_heads x dim
        attended = torch.bmm(attention.transpose(1, 2), data)
        # average over attention heads and return.
        # dimension is batchsize x dim
        return attended.mean(dim=1)


def _default_padding_mask(data: torch.Tensor) -> torch.Tensor:
    """
    Builds a 1s padding mask taking into account initial 2 dimensions
    of input data.

    Parameters
    ----------
    data : torch.Tensor
        The input data, as a tensor of shape [B x S x H]

    Returns
    ----------
    torch.Tensor
        A padding mask , as a tensor of shape [B x S]
    """
    return torch.ones((data.size(0), data.size(1))).to(data)


def _sum_with_padding_mask(data: torch.Tensor,
                           padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies padding_mask and performs summation over the data

    Parameters
    ----------
    data : torch.Tensor
        The input data, as a tensor of shape [B x S x H]
    padding_mask: torch.Tensor
        The input mask, as a tensor of shape [B X S]
    Returns
    ----------
    torch.Tensor
        The result of the summation, as a tensor of shape [B x H]

    """
    return (data * padding_mask.unsqueeze(2)).sum(dim=1)
