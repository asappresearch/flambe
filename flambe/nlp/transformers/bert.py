"""
Intergation of the pytorch_transformers bert module
"""

from typing import Tuple, Dict, Any, Optional, Union

import torch
from torch import Tensor

from flambe.compile import registrable_factory
from flambe.field import TextField
from flambe.nn import Module

import pytorch_transformers as pt


class BERTTextField(TextField, pt.BertTokenizer):
    """Perform WordPiece tokenization.

    Inspired by: https://github.com/huggingface/pytorch-pretrained-BERT/
    blob/master/pytorch_pretrained_bert/tokenization.py.

    Note that this object requires a pretrained vocabulary.

    """
    def __init__(self,  # nosec
                 vocab_file: str,
                 sos_token: str = '[CLS]',
                 eos_token: str = '[SEP]',
                 do_lower_case: bool = False,
                 max_len_truncate: int = 100, **kwargs) -> None:
        """Initialize the BERTTextField.

        Parameters
        ----------
        vocab_file : str
            Where to load the vocabulary from
        do_lower_case : bool, optional
            Set to lowercasr the input data
        max_len_truncate : int, optional
            The maximum length of a sequence
        never_split : tuple, optional
            These won't be passed to the WordPieceTokenizer

        """
        pt.BertTokenizer.__init__(self, vocab_file, do_lower_case=do_lower_case)
        self._vocab = self.vocab
        self.sos = sos_token
        self.eos = eos_token
        self.lower = do_lower_case
        self.tokenizer = self.tokenize
        self.max_len_truncate = max_len_truncate

    @registrable_factory
    @classmethod
    def from_alias(cls,
                   path: str = 'bert-base-cased',
                   cache_dir: Optional[str] = None,
                   do_lower_case: bool = False,
                   max_len_truncate: int = 100,
                   **kwargs) -> 'BERTTextField':
        """Initialize from a pretrained tokenizer.

        Parameters
        ----------
        path: str
            Path to a pretrained model, or one of the following string
            aliases currently available:
            . `bert-base-uncased`
            . `bert-large-uncased`
            . `bert-base-cased`
            . `bert-large-cased`
            . `bert-base-multilingual-uncased`
            . `bert-base-multilingual-cased`
            . `bert-base-chinese`

        """
        if 'uncased' in path and not do_lower_case:
            raise ValueError("Using uncased model but do_lower_case is False.")

        field = super().from_pretrained(path, cache_dir=cache_dir, **kwargs)
        field.basic_tokenizer.do_lower_case = do_lower_case
        field.max_len_truncate = max_len_truncate

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
        tokens = tokens[:self.max_len_truncate]

        # Add extra tokens
        if self.sos is not None:
            tokens = [self.sos] + list(tokens)
        if self.eos is not None:
            tokens = list(tokens) + [self.eos]

        numericals = self.convert_tokens_to_ids(tokens)
        return torch.tensor(numericals)


class BERTEmbeddings(Module, pt.modeling_bert.BertPreTrainedModel):
    """Integrate the pytorch_pretrained_bert BERT word embedding model.

    This module can be used as any normal encoder, or it can be
    loaded with the official pretrained BERT models. Simply used
    the `from_pretrained` class method when initializing the model.

    Currently available:
    . `bert-base-uncased`
    . `bert-large-uncased`
    . `bert-base-cased`
    . `bert-large-cased`
    . `bert-base-multilingual-uncased`
    . `bert-base-multilingual-cased`
    . `bert-base-chinese`

    """
    def __init__(self,
                 input_size_or_config: Union[int, pt.BertConfig],
                 embedding_size: int = 768,
                 embedding_dropout: float = 0.1,
                 embedding_freeze: bool = False,
                 pad_index: int = 0,
                 max_position_embeddings: int = 512,
                 type_vocab_size: int = 2, **kwargs) -> None:
        """Initialize the BERTEmbeddings.

        Parameters
        ----------
        input_size_or_config: int
            Vocabulary size of `inputs_ids` in the model's input
        hidden_size: int, optional
            Size of the encoder layers and the pooler layer.
        hidden_dropout_prob: float, optional
            The dropout probabilitiy for all fully connected layers
            in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: float, optional
            The dropout ratio for the attention probabilities.
        max_position_embeddings: int, optional
            The maximum sequence length that this model might
            ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size: int, optional
            The vocabulary size of the `token_type_ids`

        """
        Module.__init__(self)

        if isinstance(input_size_or_config, int):
            tmp_config: Dict[str, Any] = {}
            tmp_config['vocab_size'] = input_size_or_config
            tmp_config['hidden_size'] = embedding_size
            tmp_config['hidden_dropout_prob'] = embedding_dropout
            tmp_config['max_position_embeddings'] = max_position_embeddings
            tmp_config['type_vocab_size'] = type_vocab_size
            config: pt.BertConfig = pt.BertConfig(**tmp_config)
        else:
            config = input_size_or_config

        self.config = config
        self.pad_index = pad_index
        self.embeddings = pt.modeling.BertEmbeddings(config)

        self.apply(self.init_bert_weights)
        if embedding_freeze:
            for param in self.parameters():
                param.requires_grad = False

    @registrable_factory
    @classmethod
    def from_alias(cls,
                   path: str = 'bert-base-cased',
                   cache_dir: Optional[str] = None, **kwargs) -> 'BERTEmbeddings':
        """Initialize from a pretrained model.

        Parameters
        ----------
        path: str
            Path to a pretrained model, or one of the following string
            aliases currently available:
            . `bert-base-uncased`
            . `bert-large-uncased`
            . `bert-base-cased`
            . `bert-large-cased`
            . `bert-base-multilingual-uncased`
            . `bert-base-multilingual-cased`
            . `bert-base-chinese`

        """
        return super().from_pretrained(path, cache_dir=cache_dir, **kwargs)

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

        embedded = self.embeddings(data)
        return embedded, mask


class BERTEncoder(Module, pt.modeling_bert.BertPreTrainedModel):
    """Integrate the pytorch_pretrained_bert BERT encoder model.

    This module can be used as any normal encoder, or it can be
    loaded with the official pretrained BERT models. Simply used
    the `from_pretrained` class method when initializing the model.

    Currently available:
    . `bert-base-uncased`
    . `bert-large-uncased`
    . `bert-base-cased`
    . `bert-large-cased`
    . `bert-base-multilingual-uncased`
    . `bert-base-multilingual-cased`
    . `bert-base-chinese`

    """
    def __init__(self,
                 input_size_or_config: Union[int, pt.modeling_bert.BertConfig],
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 type_vocab_size: int = 2,
                 initializer_range: float = 0.02,
                 pool_last: bool = False, **kwargs) -> None:
        """Initialize the BertEncoder.

        Parameters
        ----------
        input_size_or_config: int
            Vocabulary size of `inputs_ids` in the model's input
        hidden_size: int, optional
            Size of the encoder layers and the pooler layer.
        num_hidden_layers: int, optional
            Number of hidden layers in the Transformer encoder.
        num_attention_heads: int, optional
            Number of attention heads for each attention layer in
            the Transformer encoder.
        intermediate_size: int, optional
            The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        hidden_act: str, optional
            The non-linear activation function (function or string) in
            the encoder and pooler.
            If string, "gelu", "relu" and "swish" are supported.
        hidden_dropout_prob: float, optional
            The dropout probabilitiy for all fully connected layers
            in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: float, optional
            The dropout ratio for the attention probabilities.
        max_position_embeddings: int, optional
            The maximum sequence length that this model might
            ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size: int, optional
            The vocabulary size of the `token_type_ids`
        initializer_range: float, optional
            The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.

        """
        Module.__init__(self)

        if isinstance(input_size_or_config, int):
            tmp_config: Dict[str, Any] = {}
            tmp_config['vocab_size_or_config_json_file'] = input_size_or_config
            tmp_config['hidden_size'] = hidden_size
            tmp_config['num_hidden_layers'] = num_hidden_layers
            tmp_config['num_attention_heads'] = num_attention_heads
            tmp_config['hidden_act'] = hidden_act
            tmp_config['intermediate_size'] = intermediate_size
            tmp_config['hidden_dropout_prob'] = hidden_dropout_prob
            tmp_config['attention_probs_dropout_prob'] = attention_probs_dropout_prob
            tmp_config['max_position_embeddings'] = max_position_embeddings
            tmp_config['type_vocab_size'] = type_vocab_size
            tmp_config['initializer_range'] = initializer_range
            config: pt.BertConfig = pt.BertConfig(**tmp_config)
        else:
            config = input_size_or_config

        self.config = config
        self.encoder = pt.modeling_bert.BertEncoder(config)
        self.pooler = pt.modeling_bert.BertPooler(config)

        self.pool_last = pool_last

        self.apply(self.init_bert_weights)

    @registrable_factory
    @classmethod
    def from_alias(cls,
                   path: str = 'bert-base-cased',
                   cache_dir: Optional[str] = None,
                   pool_last: bool = False, **kwargs) -> 'BERTEncoder':
        """Initialize from a pretrained model.

        Parameters
        ----------
        path: str
            Path to a pretrained model, or one of the following string
            aliases currently available:
            . `bert-base-uncased`
            . `bert-large-uncased`
            . `bert-base-cased`
            . `bert-large-cased`
            . `bert-base-multilingual-uncased`
            . `bert-base-multilingual-cased`
            . `bert-base-chinese`

        """
        model = super().from_pretrained(path, cache_dir=cache_dir, **kwargs)
        model.pool_last = pool_last

        return model

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
        attention_mask = mask if mask is not None else torch.ones_like(data)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular
        # masking of causal attention used in OpenAI GPT, we just need
        # to prepare the broadcastdimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend
        # and 0.0 for masked positions, this operation will create a
        # tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax,
        # this is effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        # Write (1.0 - extended_attention_mask) weird way for mypy
        # so it uses Tensor.__add__ instead of float's
        extended_attention_mask = (-extended_attention_mask + 1.0) * -10000.0

        encoded_layers = self.encoder(data, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        out = pooled_output if self.pool_last else sequence_output
        return out
