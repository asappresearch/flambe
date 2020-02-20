import os
import subprocess
import sys
from tempfile import NamedTemporaryFile as tmpfile
from tempfile import TemporaryDirectory as tmpdir

import flambe as fl


def module_equals(model1, model2):
    """Check if 2 pytorch modules have the same weights

    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_exporter_builder(top_level):
    with tmpdir() as d, tmpdir() as d2, tmpfile(mode="w", suffix=".yaml") as f, tmpfile(mode="w", suffix=".yaml") as f2:
        # First run an experiment
        exp = """
!Experiment

pipeline:
  dataset: !TabularDataset.from_path
    train_path: {top_level}/tests/data/dummy_tabular/train.csv
    val_path: {top_level}/tests/data/dummy_tabular/val.csv
    sep: ','
    transform:
      text: !TextField
      label: !LabelField
  model: !TextClassifier
    embedder: !Embedder
      embedding: !torch.nn.Embedding
        num_embeddings: !copy dataset.text.vocab_size
        embedding_dim: 30
      encoder: !PooledRNNEncoder
        input_size: 30
        rnn_type: lstm
        n_layers: 1
        hidden_size: 16
    output_layer: !SoftmaxLayer
      input_size: !ref model[embedder].encoder.rnn.hidden_size
      output_size: !copy dataset.label.vocab_size

  exporter: !Exporter
    model: !copy model
    text: !copy dataset.text
"""

        exp = exp.format(top_level=top_level)
        f.write(exp)
        f.flush()
        ret = subprocess.run(['flambe', 'run', f.name, '-o', d])
        assert ret.returncode == 0

        # Then run a builder

        builder = """
!Environment
extensions:
  flambe_inference: {top_level}/tests/data/dummy_extensions/inference/
---

!Builder

obj: !flambe_inference.DummyInferenceEngine
  model: !torch.load
    f: {path}
"""
        model_path = os.path.join(d, "flambe_output", "exporter", '0', 'checkpoint.pt')

        builder = builder.format(path=model_path, top_level=top_level)
        f2.write(builder)
        f2.flush()

        ret = subprocess.run(['flambe', 'run', f2.name, '-o', d2])
        assert ret.returncode == 0

        # Import the module after import_modules (which registered tags already)
        sys.path.append(f'{top_level}/tests/data/dummy_extensions/inference/')
        from flambe_inference import DummyInferenceEngine

        eng1 = fl.load(os.path.join(d2, 'flambe_output', 'checkpoint.pt'))

        assert type(eng1) is DummyInferenceEngine
        assert type(eng1.model.model) is fl.nlp.classification.TextClassifier

        # extension_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests/data/dummy_extensions/inference")
        # assert eng1._extensions == {"flambe_inference": extension_path}

        # Revisit this after changes to serializations
        # eng2 = DummyInferenceEngine.load_from_path(d2)

        # assert type(eng2) is DummyInferenceEngine
        # assert type(eng2.model) is TextClassifier

        # assert eng2._extensions == {"flambe_inference": extension_path}

        # assert module_equals(eng1.model, eng2.model)
