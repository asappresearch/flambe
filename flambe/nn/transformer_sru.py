from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from sru import SRUCell

from flambe.nn import Module
from flambe.nn.transformer import Transformer, TransformerEncoder, TransformerDecoder, _get_clones


class TransformerSRU(Transformer):
    """A Transformer with an SRU replacing the FFN."""

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 bidrectional: bool = False,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize the TransformerSRU Model.

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
        bidrectional: bool, optional
            Whether the SRU Encoder module should be bidrectional.
            Defaul ``False``.

        Extra keyword arguments are passed to the SRUCell.

        """
        super(Transformer, self).__init__()

        self.encoder = TransformerSRUEncoder(d_model,
                                             nhead,
                                             dim_feedforward,
                                             num_encoder_layers,
                                             dropout,
                                             bidrectional,
                                             **kwargs)

        self.decoder = TransformerSRUDecoder(d_model,
                                             nhead,
                                             dim_feedforward,
                                             num_encoder_layers,
                                             dropout,
                                             **kwargs)


class TransformerSRUEncoder(TransformerEncoder):
    """A TransformerSRUEncoder with an SRU replacing the FFN."""

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize the TransformerEncoder.

        Parameters
        ---------
        d_model : int
            the number of expected features in encoder/decoder inputs.
            Default ``512``.
        nhead : int, optional
            the number of heads in the multiheadattention
            Default ``8``.
        num_layers : int
            the number of sub-encoder-layers in the encoder (required).
            Default ``6``.
        dim_feedforward : int, optional
            the inner feedforard dimension. Default ``2048``.
        dropout : float, optional
            the dropout percentage. Default ``0.1``.
        bidirectional: bool
            Whether the SRU module should be bidrectional.
            Defaul ``False``.

        Extra keyword arguments are passed to the SRUCell.

        """
        encoder_layer = TransformerSRUEncoderLayer(d_model,
                                                   nhead,
                                                   dim_feedforward,
                                                   dropout,
                                                   bidirectional)

        self.layers = _get_clones(encoder_layer, num_layers)
        self._reset_parameters()


class TransformerSRUDecoder(TransformerDecoder):
    """A TransformerSRUDecoderwith an SRU replacing the FFN."""

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize the TransformerEncoder.

        Parameters
        ---------
        d_model : int
            the number of expected features in encoder/decoder inputs.
            Default ``512``.
        nhead : int, optional
            the number of heads in the multiheadattention
            Default ``8``.
        num_layers : int
            the number of sub-encoder-layers in the encoder (required).
            Default ``6``.
        dim_feedforward : int, optional
            the inner feedforard dimension. Default ``2048``.
        dropout : float, optional
            the dropout percentage. Default ``0.1``.

        Extra keyword arguments are passed to the SRUCell.

        """
        decoder_layer = TransformerSRUDecoderLayer(d_model,
                                                   nhead,
                                                   dim_feedforward,
                                                   dropout)

        self.layers = _get_clones(decoder_layer, num_layers)
        self._reset_parameters()


class TransformerSRUEncoderLayer(Module):
    """A TransformerSRUEncoderLayer with an SRU replacing the FFN."""

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize a TransformerSRUEncoderLayer.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        n_head : int
            The number of heads in the multiheadattention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network (default=2048).
        dropout : float, optional
            The dropout value (default=0.1).
        bidirectional: bool
            Whether the SRU module should be bidrectional.
            Defaul ``False``.

        Extra keyword arguments are passed to the SRUCell.

        """
        super(TransformerSRUEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.sru = SRUCell(d_model, dim_feedforward, dropout, bidirectional=bidirectional, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,  # type: ignore
                src: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass the input through the endocder layer.

        Parameters
        ----------
        src: torch.Tensor
            The seqeunce to the encoder layer (required).
        memory: Optional[torch.Tensor]
            Optional memory from previous sequence.
        src_mask: torch.Tensor, optional
            The mask for the src sequence (optional).
        src_key_padding_mask: torch.Tensor, optional
            The mask for the src keys per batch (optional).

        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.sru(src, state=memory, mask_pad=src_key_padding_mask))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerSRUDecoderLayer(Module):
    """A TransformerSRUDecoderLayer with an SRU replacing the FFN."""

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize a TransformerDecoder.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        n_head : int
            The number of heads in the multiheadattention models.
        dim_feedforward : int, optional
            The dimension of the feedforward network (default=2048).
        dropout : float, optional
            The dropout value (default=0.1).

        Extra keyword arguments are passed to the SRUCell.

        """
        super(TransformerSRUDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.sru = SRUCell(d_model, dim_feedforward, dropout, bidirectional=False, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,  # type: ignore
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Parameters
        ----------
        tgt: torch.Tensor
            The sequence to the decoder layer (required).
        memory: torch.Tensor
            The sequnce from the last layer of the encoder (required).
        tgt_mask: torch.Tensor, optional
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor, optional
            the mask for the memory sequence (optional).
        tgt_key_padding_mask: torch.Tensor, optional
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask: torch.Tensor, optional
            the mask for the memory keys per batch (optional).

        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.sru(tgt, state=memory, mask_pad=tgt_key_padding_mask))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
