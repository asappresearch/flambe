
from typing import Tuple, Union, Optional

from torch import nn
from torch import Tensor

from flambe.compile import registrable_factory
from flambe.nn.module import Module


class Embeddings(Module, nn.Embedding):
    """Implement an Embedding module.

    This object replicates the usage of nn.Embedding but
    registers the from_pretrained classmethod to be used inside
    a Flambé configuration, as this does not happen automatically
    during the registration of PyTorch objects.

    """

    @registrable_factory
    @classmethod
    def from_pretrained(cls,
                        embeddings: Tensor,
                        freeze: bool = True,
                        paddinx_idx: Optional[int] = None,
                        max_norm: Optional[float] = None,
                        norm_type: float = 2.0,
                        scale_grad_by_freq: bool = False,
                        sparse: bool = False):
        """Create Embedding instance from given 2-dimensional Tensor.

        Parameters
        ----------
        embeddings: torch.Tensor
            FloatTensor containing weights for the Embedding.
            First dimension is being passed to Embedding as
            num_embeddings, second as embedding_dim.
        freeze: bool
            If True, the tensor does not get updated in the learning
            process. Default: True
        padding_idx (int, optional)
            See module initialization documentation.
        max_norm: float, optional
            See module initialization documentation.
        norm_type: float, optional
            See module initialization documentation. Default 2.
        scale_grad_by_freq: bool, optional
            See module initialization documentation. Default False.
        sparse (bool, optional)
            See module initialization documentation. Default False.

        """
        return super().from_pretrained(embeddings,
                                       freeze,
                                       paddinx_idx,
                                       max_norm,
                                       norm_type,
                                       scale_grad_by_freq,
                                       sparse)


class Embedder(Module):
    """Implements an Embedder module.

    An Embedder takes as input a sequence of index tokens,
    and computes the corresponding embedded representations, and
    padding mask. The encoder may be initialized using a pretrained
    embedding matrix.

    Attributes
    ----------
    embeddings: Embedding
        The embedding layer
    encoder: Encoder
        The sub-encoder that this object is wrapping
    drop: nn.Dropout
        The dropout layer

    """
    def __init__(self,
                 embedding: nn.Embedding,
                 encoder: Module,
                 embedding_dropout: float = 0,
                 pad_index: Optional[int] = 0) -> None:
        """Initializes the TextEncoder module.

        Extra arguments are passed to the nn.Embedding module.

        Parameters
        ----------
        embedding: nn.Embedding
            The embedding layer
        encoder: Module
            The encoder
        embedding_dropout: float, optional
            Amount of dropout between the embeddings and the encoder
        pad_index: int, optional
            Passed the nn.Embedding object. See pytorch documentation.

        """
        super().__init__()

        self.embedding = embedding
        self.dropout = nn.Dropout(embedding_dropout)
        self.encoder = encoder
        self.pad_index = pad_index

    def forward(self, data: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor, batch first

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]]
            The encoded output, as a float tensor. May return a state
            if the encoder is an RNN

        """
        embedded = self.embedding(data)
        embedded = self.dropout(embedded)

        if self.pad_index is not None:
            mask = (data != self.pad_index).float()
            encoding = self.encoder(embedded, mask=mask)
        else:
            encoding = self.encoder(embedded)

        return encoding
