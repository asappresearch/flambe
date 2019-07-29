"""
Intergation of the pytorch_transformers openai module
"""

import copy
from typing import Dict, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from flambe.compile import registrable_factory
from flambe.nn import Module
from flambe.field import TextField

import pytorch_transformers as pt


class OpenAIGPTTextField(TextField, pt.OpenAIGPTTokenizer):
    """Perform WordPiece tokenization.

    Inspired by: https://github.com/huggingface/pytorch-pretrained-BERT/
    blob/master/pytorch_pretrained_bert/tokenization_openai.py.

    Note that this object requires a pretrained vocabulary.

    """
    def __init__(self,
                 vocab_file: str,
                 merges_file: str,
                 max_len: int = 100,
                 lower: bool = False) -> None:
        """Initialize the BERTTextField.

        Parameters
        ----------
        vocab_file : str
            Where to load the vocabulary from
        merges_file : str
            Where to load the wordpiece splits from

        """
        pt.OpenAIGPTTokenizer.__init__(self, vocab_file, merges_file)
        self.lower = lower
        self._vocab = self.encoder

    @registrable_factory
    @classmethod
    def from_alias(cls,
                   path: str = 'openai-gpt',
                   cache_dir: Optional[str] = None) -> 'OpenAIGPTTextField':
        """Initialize from a pretrained tokenizer.
        """
        field = super().from_pretrained(path, cache_dir=cache_dir)
        return field

    def process(self, example: str) -> torch.Tensor:  # type: ignore
        """Process an example, and create a Tensor.

        Parameters
        ----------
        example: str
            The example to process, as a single string

        Returns
        -------
        torch.Tensor
            The processed example, tokenized and numericalized

        """
        tokens = self.tokenize(example)
        tokens = tokens[:self.max_len]

        numericals = self.convert_tokens_to_ids(tokens)
        return torch.tensor(numericals)


class OpenAIGPTEmbeddings(Module, pt.modeling_openai.OpenAIGPTPreTrainedModel):
    """Integrate the pytorch_pretrained_bert OpenAI embedding model.

    This module can be used as any normal encoder, or it can be
    loaded with the official pretrained OpenAI models. Simply used
    the `from_pretrained` class method when initializing the model.

    """
    def __init__(self,
                 input_size_or_config: Union[int, pt.OpenAIGPTConfig] = 40478,
                 embedding_size: int = 768,
                 embedding_dropout: float = 0.1,
                 embedding_freeze: bool = False,
                 pad_index: int = 0,
                 n_special: int = 0,
                 n_positions: int = 512,
                 initializer_range=0.02) -> None:
        """Initialize the OpenAIGPTEmbeddings.

        Parameters
        ----------
        input_size_or_config: int
            Vocabulary size or configuration
        n_special: int
            The number of special tokens to learn
            during fine-tuning ('[SEP]', '[CLF]', ...)
        n_positions: int, optional
            Number of positional embeddings.
        embedding_size: int, optional
            Dimensionality of the embeddings and hidden states.
        embedding_dropout: float, optional
            The dropout ratio for the embeddings.
        initializer_range: float, optional
            The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.

        """
        Module.__init__(self)

        if isinstance(input_size_or_config, int):
            tmp_config: Dict[str, Any] = {}
            tmp_config['vocab_size_or_config_json_file'] = input_size_or_config
            tmp_config['n_embd'] = embedding_size
            tmp_config['embd_pdrop'] = embedding_dropout
            tmp_config['n_special'] = n_special
            tmp_config['n_positions'] = n_positions
            self.config: pt.OpenAIGPTConfig = pt.OpenAIGPTConfig(**tmp_config)
        else:
            self.config = input_size_or_config

        num_tokens = self.config.vocab_size + self.config.n_special
        self.tokens_embed = nn.Embedding(num_tokens, self.config.n_embd)
        self.positions_embed = nn.Embedding(self.config.n_positions, self.config.n_embd)
        self.drop = nn.Dropout(self.config.embd_pdrop)

        self.apply(self.init_weights)

        if embedding_freeze:
            for param in self.parameters():
                param.requires_grad = False

    @registrable_factory
    @classmethod
    def from_alias(cls,
                   path: str = 'openai-gpt',
                   cache_dir: Optional[str] = None) -> 'OpenAIGPTEmbeddings':
        """Initialize from a pretrained model.

        Parameters
        ----------
        path: str
            Path to a pretrained model, or one of the following string
            aliases currently available:
            . `openai-gpt`

        """
        return super().from_pretrained(path, cache_dir=cache_dir)

    def set_num_special_tokens(self, num_special_tokens):
        " Update input embeddings with new embedding matrice if needed "
        return super().set_num_special_tokens(num_special_tokens)

    def forward(self, data: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor, batch first

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor, batch_first
        torch.Tensor, optional
            The padding mask if a pad index was given

        """
        mask = None
        if self.pad_index is not None:
            mask = (data != self.pad_index).float()
        # This was used when we had a single embedding
        # matrice from position and token embeddings
        # start = self.config.vocab_size + self.config.n_special
        # end = start + data.size(-1)
        # position_ids = torch.arange(start, end,
        #                dtype=torch.long, device=data.device)
        position_ids = torch.arange(data.size(-1), dtype=torch.long, device=data.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data)

        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.tokens_embed(data)
        position_embeds = self.positions_embed(position_ids)
        token_type_embeds = 0
        # Add the position information to the input embeddings
        # h = e.sum(dim=2)
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        return hidden_states, mask


class OpenAIGPTEncoder(Module, pt.modeling_openai.OpenAIGPTPreTrainedModel):
    """Integrate the pytorch_pretrained_bert OpenAIGPT encoder model.

    This module can be used as any normal encoder, or it can be
    loaded with the official pretrained BERT models. Simply used
    the `from_pretrained` class method when initializing the model.

    Currently available:
    . `openai-gpt`

    """
    def __init__(self,
                 input_size_or_config: Union[int, pt.OpenAIGPTConfig] = 768,
                 n_ctx: int = 512,
                 n_layer: int = 12,
                 n_head: int = 12,
                 afn: Union[str, nn.Module] = "gelu",
                 resid_pdrop: float = 0.1,
                 embd_pdrop: float = 0.1,
                 attn_pdrop: float = 0.1,
                 layer_norm_epsilon: float = 1e-5,
                 initializer_range=0.02) -> None:
        """Initialize the OpenAIGPTEncoder.

        Parameters
        ----------
        input_size_or_config: Union[int, OpenAIGPTConfig]
            Vocabulary size or configuration
        n_ctx: int
            Size of the causal mask (usually same as n_positions).
        n_layer: int, optional
            Number of hidden layers in the Transformer encoder.
        n_head: int, optional
            Number of attention heads for each attention layer in
            the Transformer encoder.
        afn: Union[str, nn.Module]
            The non-linear activation function (module or string) in
            the encoder and pooler. If string, "gelu", "relu" and
            "swish" are supported.
        resid_pdrop: float, optional
            The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        attn_pdrop: float, optional
            The dropout ratio for the attention probabilities.
        layer_norm_epsilon: float, optional
            epsilon to use in the layer norm layers
        initializer_range: float, optional
            The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.

        """
        Module.__init__(self)

        if isinstance(input_size_or_config, int):
            tmp_config: Dict[str, Any] = {}

            tmp_config['n_embd'] = input_size_or_config
            tmp_config['n_ctx'] = n_ctx
            tmp_config['n_layer'] = n_layer
            tmp_config['n_head'] = n_head
            tmp_config['afn'] = afn
            tmp_config['resid_pdrop'] = resid_pdrop
            tmp_config['attn_pdrop'] = attn_pdrop
            tmp_config['layer_norm_epsilon'] = layer_norm_epsilon
            tmp_config['initializer_range'] = initializer_range
            self.config: pt.OpenAIGPTConfig = pt.OpenAIGPTConfig(**tmp_config)
        else:
            self.config = input_size_or_config

        block = pt.modeling_openai.Block(self.config.n_ctx, self.config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(self.config.n_layer)])

        self.apply(self.init_weights)

    @registrable_factory
    @classmethod
    def from_alias(cls,
                   path: str = 'openai-gpt',
                   cache_dir: Optional[str] = None) -> 'OpenAIGPTEncoder':
        """Initialize from a pretrained model.

        Parameters
        ----------
        path: str
            Path to a pretrained model, or one of the following string
            aliases currently available:
            . `openai-gpt`

        """
        return super().from_pretrained(path, cache_dir=cache_dir)

    def forward(self,
                data: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a long tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor or the pooled output

        """
        for block in self.h:
            hidden_states = block(hidden_states)

        return hidden_states
