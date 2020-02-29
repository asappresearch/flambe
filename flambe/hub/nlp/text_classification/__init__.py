# type: ignore[attr-define]

from flambe.hub.nlp.text_classification.datasets import SSTDataset, TRECDataset, NewsGroupDataset
from flambe.hub.nlp.text_classification.model import TextClassifier


__all__ = ['TextClassifier', 'SSTDataset', 'TRECDataset', 'NewsGroupDataset']
