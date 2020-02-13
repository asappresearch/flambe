from flambe import Component
from flambe.nlp.classification import TextClassifier


class DummyInferenceEngine(Component):

    def __init__(self, model: TextClassifier) -> None:
        self.model = model

    def run(self):
        pass
