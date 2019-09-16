from typing import Optional

import torch

from flambe.nn import Module


def is_mask_valid(padding_mask: torch.Tensor, data_size: torch.Size) -> bool:
    """Return if a mask is valid.

    Parameters
    ----------
    padding_mask: torch.Tensor
        The padding mask.
    data_size: torch.Size
        The data size where the mask will be applied.

    Returns
    -------
    bool
        If the mask is valid.

    """
    # Check that mask is all 0s and 1s
    if not torch.all((padding_mask == 0).byte() | (padding_mask == 1).byte()):
        return False

    # Check that mask size match the data size
    if padding_mask.size() != data_size[0:2]:
        return False

    # TODO -> add check to see if mask contains first 1s and trailing 0s
    return True


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
        if data.size(1) == 0:
            raise ValueError("Can't pool value from empty sequence")

        if padding_mask is not None and not is_mask_valid(padding_mask, data.size()):
            raise ValueError(f"Padding mask is not valid. It should contain only 1s and " +
                             "trailing 0s, but instead it received {padding_mask}")

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
        if data.size(1) == 0:
            raise ValueError("Can't pool value from empty sequence")

        # Compute lengths
        if padding_mask is None:
            lengths = torch.tensor([data.size(1)] * data.size(0)).long()
        else:
            if not is_mask_valid(padding_mask, data.size()):
                raise ValueError(f"Padding mask is not valid. It should contain only 1s and " +
                                 "trailing 0s, but instead it received {padding_mask}")

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
        if data.size(1) == 0:
            raise ValueError("Can't pool value from empty sequence")

        # Apply pooling
        if padding_mask is None:
            padding_mask = torch.ones((data.size(0), data.size(1))).to(data)

        if not is_mask_valid(padding_mask, data.size()):
            raise ValueError(f"Padding mask is not valid. It should contain only 1s and " +
                             "trailing 0s, but instead it received {padding_mask}")

        return (data * padding_mask.unsqueeze(2)).sum(dim=1)


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
        if data.size(1) == 0:
            raise ValueError("Can't pool value from empty sequence")

        # Apply pooling
        if padding_mask is None:
            padding_mask = torch.ones((data.size(0), data.size(1))).to(data)

        if not is_mask_valid(padding_mask, data.size()):
            raise ValueError(f"Padding mask is not valid. It should contain only 1s and " +
                             "trailing 0s, but instead it received {padding_mask}")

        data = (data * padding_mask.unsqueeze(2)).sum(dim=1)
        return data / padding_mask.sum(dim=1).unsqueeze(1)
