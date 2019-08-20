"""
Code taken from the PyTorch source code. Slightly modified to improve
the interface to the TransformerEncoder, and TransformerDecoder modules.

"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flambe.nn import Module

from nn.activation import MultiheadAttention
from nn.container import ModuleList
from nn.init import xavier_uniform_


class Transformer(Module):
    r"""A Transformer model

    User is able to modify the attributes as needed. The architechture
    is based on the paper "Attention Is All You Need". Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
    Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    """

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 custom_encoder: Optional[Module] = None,
                 custom_decoder: Optional[Module] = None) -> None:
        """Initialize the Transformer Model.

        Parameters
        ----------
        d_model : int, optional
            the number of expected features in the
            encoder/decoder inputs (default=512).
        nhead : int, optional
            the number of heads in the multiheadattention
            models (default=8).
        num_encoder_layers : int, optional
            the number of sub-encoder-layers in the encoder
            (default=6).
        num_decoder_layers : int, optional
            the number of sub-decoder-layers in the decoder
            (default=6).
        dim_feedforward : int, optional
            the dimension of the feedforward network model
            (default=2048).
        dropout : float, optional
            the dropout value (default=0.1).
        custom_encoder : [type], optional
            custom encoder (default=None).
        custom_decoder : [type], optional
            custom decoder (default=None).

        """
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = TransformerEncoder(d_model,
                                              nhead,
                                              dim_feedforward,
                                              num_encoder_layers,
                                              dropout=dropout,
                                              norm=True)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            self.decoder = TransformerDecoder(d_model,
                                              nhead,
                                              dim_feedforward,
                                              num_encoder_layers,
                                              dropout=dropout,
                                              norm=True)

    def forward(self,  # type: ignore
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Take in and process masked source/target sequences.

        Parameters
        ----------
        src: torch.Tensor
            the sequence to the encoder (required).
            shape: :math:`(S, N, E)`.
        tgt: torch.Tensor
            the sequence to the decoder (required).
            shape: :math:`(T, N, E)`.
        src_mask: Optional[torch.Tensor]
            the additive mask for the src sequence (optional).
            shape: :math:`(S, S)`.
        tgt_mask: Optional[torch.Tensor]
            the additive mask for the tgt sequence (optional).
            shape: :math:`(T, T)`.
        memory_mask: Optional[torch.Tensor]
            the additive mask for the encoder output (optional).
            shape: :math:`(T, S)`.
        src_key_padding_mask: Optional[torch.Tensor]
            the ByteTensor mask for src keys per batch (optional).
            shape: :math:`(N, S)`
        tgt_key_padding_mask: Optional[torch.Tensor]
            the ByteTensor mask for tgt keys per batch (optional).
            shape: :math:`(N, T)`.
        memory_key_padding_mask: Optional[torch.Tensor]
            the ByteTensor mask for memory keys per batch (optional).
            shape" :math:`(N, S)`.

        Returns
        -------
        output: torch.Tensor
            The output sequence, shape: :math:`(T, N, E)`.

        Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else.
            These masks ensure that predictions for position i depend
            only on the unmasked positions j and are applied identically
            for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor
            where True values are positions that should be masked with
            float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from
            position i if it is masked, and has a separate mask for each
            sequence in a batch.
        Note: Due to the multi-head attention architecture in the
            transformer model, the output sequence length of a
            transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target
            sequence length, N is the batchsize, E is the feature number

        """

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence.

        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: 
        norm: the layer normalization component (optional).

    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        """Initialize the TransformerEncoder.

        Parameters
        ----------
        d_model : int
            [description]
        nhead : int
            [description]
        num_layers : int
            the number of sub-encoder-layers in the encoder (required).
        dim_feedforward : int, optional
            [description], by default 2048
        dropout : float, optional
            [description], by default 0.1

        """
        super(TransformerEncoder, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model) if norm else None

        self._reset_parameters()

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    r"""A TransformerDecoderLayer.
    
    A TransformerDecoderLayer is made up of self-attn, multi-head-attn
    and feedforward network. This standard decoder layer is based on the
    paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz
    Kaiser, and Illia Polosukhin. 2017. Attention is all you need.
    In Advances in Neural Information Processing Systems,
    pages 6000-6010. Users may modify or implement in a different way
    during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])