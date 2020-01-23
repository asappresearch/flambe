from typing import Dict, Optional, Any

import torch.nn as nn

from flambe.dataset import TabularDataset
from flambe.sampler import BaseSampler
from flambe.nn import Module, Encoder, Embedder, Embeddings
from flambe.metric import Accuracy
from flambe.field import TextField, LabelField
from flambe.task import Training
from flambe.optim.optimizer import Optimizer
from flambe.optim.lr_scheduler import LRScheduler


class TextClassification(Task):
    """A text classification task.

    Performs text classifcation training and evaluation.
    Takes as input a dataset and an encoder, and constructs
    a simple TextClassifier. You may pass your custom fields
    or used the defaults. The loss is computer via a cross entropy,
    and the validation metric is accuracy.

    """

    def __init__(self,
                 dataset: TabularDataset,
                 encoder: Optional[Encoder] = None,
                 pooling: Optional[Module] = None,
                 dropout: float = 0,
                 build_vocabularies: bool = True,
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 embedding_args: Dict[str, Any] = None,
                 optimizer: Optional[Optimizer] = None,
                 iter_scheduler: Optional[LRScheduler] = None,
                 eval_scheduler: Optional[LRScheduler] = None,
                 text_field: Optional[TextField] = None,
                 label_field: Optional[LabelField] = None,
                 embedder: Optional[Embedder] = None,
                 model: Optional[TextClassifier] = None,
                 **kwargs) -> None:
        """Initalize a TextClassification task.

        Parameters
        ----------
        dataset : TabularDataset
            The input dataset
        encoder : Encoder
            An encoder
        pooling : Optional[Module], optional
            A pooling method
        dropout : float, optional
            Dropout to apply between the encoder and output layer.
            Default ``0``.
        build_vocabularies : bool, optional
            Whether the fields should expand their vocabulary Using
            the training data. Default ``True``.
        train_batch_size : int, optional
            The batch size to use during. Default ``32``.
        val_batch_size : int, optional
            The batch size to use during evaluation. Default ``32``.
        embedding_args : Dict[str, Any], optional
            Keyword arguments to pass the ``Embeddings`` constructor.
        optimizer : Optional[Optimizer], optional
            The optimizer to use. Should be provided for training.
        iter_scheduler : Optional[LRScheduler], optional
            A learning rate scheduler to call on every training step.
        eval_scheduler : Optional[LRScheduler], optional
            A learning rate scheduler to call on every validation step.
        text_field : Optional[TextField], optional
            A custom text field to apply to the text inputs.
        label_field : Optional[LabelField], optional
            A custom label field to apply to the label inputs.
        embedder : Optional[Embedder], optional
            A custom embedder. Overrides ``encoder`` and ``pooling``.
        model : Optional[TextClassifier], optional
            A custom model. Overrides ``encoder`` and ``embedder``.

        See the ``Training`` parent class for other keyword arguments.

        """
        super().__init__(**kwargs)

        text_field = text_field or TextField()
        label_field = label_field or LabelField()

        # Build vocabularies
        if build_vocabularies:
            text, label = zip(*dataset)
            text_field.setup(text)
            label_field.setup(label)

        self.embedding_args = embedding_args or dict()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.dataset = dataset
        self.text_field = text_field
        self.label_field = label_field
        self.dataset._set_transforms({'text': text_field, 'label': label_field}, do_setup=False)

        if encoder is None and embedder is None and model is None:
            raise ValueError("At least one of encoder, embedder or model must be provided.")
        if model is None and embedder is None and encoder is not None and pooling is None:
            raise ValueError("Must provide a pooling stratgey.")

        self.encoder = encoder
        self.pooling = pooling
        self.embedder = embedder
        self._model = model
        self.dropout = dropout
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.metric_fn = Accuracy()

        self.optimizer = optimizer
        self.eval_scheduler = eval_scheduler
        self.iter_scheduler = iter_scheduler

    def build_model(self):
        """Build the model, containing all parameters to train."""
        if self._model is not None:
            return self._model

        output_size = self.label_field.vocab_size
        if self.embedder is None:
            input_size = self.text_field.vocab_size
            embedding_dim = self.encoder.input_dim
            padding_idx = self.text_field.vocab[self.text_field.pad]
            embeddings = Embeddings(input_size, embedding_dim, padding_idx, **self.embedding_args)
            self.embedder = Embedder(embeddings, self.encoder, self.pooling, self.embedding_dropout)
        return TextClassifier(self.embedder, output_size, self.dropout)

    def build_optimizers(self, model):
        """Build the model, containing all parameters to train."""
        if self.optimizer is None:
            raise ValueError("Using the task in training mode but not optimizer was provided.")
        self.optimizer.initialize(model)
        return {'optimizer': self.optimizer}

    def build_schedulers(self, optimizers, mode='iter'):
        """Build the model, containing all parameters to train."""
        schedulers = dict()
        if self.iter_scheduler is not None and mode == 'iter':
            self.iter_scheduler.initialize(optimizers['optimizer'])
            schedulers['iter_scheduler'] = self.iter_scheduler
        elif self.eval_scheduler is not None and mode == 'eval':
            self.eval_scheduler.initialize(optimizers['optimizer'])
            schedulers['eval_scheduler'] = self.eval_scheduler
        return schedulers

    def dataloaders(self):
        """Get an iterable of batches of data."""
        batch_size = self.train_batch_size if train else self.val_batch_size
        return BaseSampler(getattr(dataset, split), suffle=train, batch_size=batch_size)

    def train_step(self, model, batch):
        """Compute loss on the given batch during training."""
        text, labels = batch
        preds = model(text.to(self.device))
        loss = self.loss_fn(preds, labels.to(self.device))
        return loss

    def val_step(self, model, batches):
        """Compute metrics on the validation set, given in batches."""
        total_loss, total_acc, total_count = 0, 0, 0
        for batch in batches:
            text, labels = batch
            preds = model(text.to(self.device))
            total_count += preds.size(0)
            total_loss += self.loss_rn(preds, labels.to(self.device)).sum().item()
            total_acc += self.metric_fn(preds, labels).sum().item()
        loss = total_loss / total_count
        accuracy = total_acc / total_count
        return {'loss': loss, 'accuracy': accuracy}

    def val_metric(self, metrics):
        return metrics['accuracy']