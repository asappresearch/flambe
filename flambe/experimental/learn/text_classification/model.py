from flambe.nn import Module, Embedder
import torch.nn as nn


class TextClassifier(Module):

    def __init__(self, embedder: Embedder, output_size: int, dropout: float = 0) -> None:
        """A simple text classifier."""
        super().__init__()
        self.output_size = output_size
        self.embedder = embedder
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedder.output_dim, output_size)

    def forward(self, data):  # type: ignore
        """Run a forward pass from shape (B x S) to (B x S x H)."""
        return self.output_layer(self.drop(self.embedder(data)))
