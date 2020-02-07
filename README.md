![Flambe Logo](imgs/Flambe_Logo_CMYK_FullColor.png)

------------

[![Fast Tests Status](https://github.com/asappresearch/flambe/workflows/tests-fast/badge.svg)](https://github.com/asappresearch/flambe/actions)

[![Slow Tests Status](https://github.com/asappresearch/flambe/workflows/tests-slow/badge.svg)](https://github.com/asappresearch/flambe/actions)

[![Documentation Status](https://readthedocs.org/projects/flambe/badge/?version=latest)](https://flambe.ai/en/latest/?badge=latest)


Welcome to Flambé, a `PyTorch <https://pytorch.org/>`_-based library that aims to abstract away
the boilerplate code tradtionally involved in machine learning research. Flambé does not reinvent
the wheel, but instead connects the dots between a curated set of frameworks. In doing so,
Flambe's core area of focus is to create the best possible user experience. With Flambé you can:

* Automate the boilerplate code in training models with PyTorch
* **Run hyperparameter searches** over arbitrary Python objects
* Constuct experiment DAGs, which include searching over hyperparameters and reducing to the
best variants at each node in the DAG.
* Execute tasks **remotely** and **in parallel** over many workers, including full AWS, GCP, and Kubernetes integration
* Easily share experiment configurations, results, and model weights with others


# Installation

**From** ``PIP``:

```bash
pip install flambe
```

**From source**:

```bash
git clone git@github.com:asappresearch/flambe.git
pip install ./flambe
```

# Getting started

There are a few core objects that offer an entrypoint to using Flambé:

| Object | Role |
| -------|------|
| Trainer | Train a model on a given task |
| Schema | Replace any keyword argument with a distirbution |
| Search | Run a hyperparameter search on a schema |
| Experiment | Construct a DAG, with a hyperameter search at each node |

## Train a model

```python
import flambe as fl

dataset = fl.nlp.SSTDataset()
model = fl.nlp.TextClassifier(
    n_layers=2
)
trainer = fl.learn.Trainer(
    dataset=dataset,
    model=model
)

# Execute training
trainer.run()
```

## Run a hyperparameter search


```python
import flambe as fl

# You can create a schema over any python object
dataset = fl.schema(fl.nlp.SSTDataset)
model = fl.schema(fl.nlp.TextClassifier)(
    # Replace any argument with a distibution
    n_layers=fl.choice([1, 2, 3])  
)
trainer = fl.schema(fl.learn.Trainer)(
    # Schemas take other schemas as inputs
    dataset=dataset,
    model=model
)

# Run a hyperparameter search
algorithm = fl.RandomSearch(max_steps=10, trial_budget=2)
search = Search(trainer, algorithm)
search.run()
```

**Note**: ``Search`` expects a schema that is wrapping an object of
the type ``Searchable``. ``Searchable`` is any object that implements
the following interface:

```python
class Searchable:

    def step(self) -> bool:
    """Indicate whether execution is complete."""
        pass
    
    def metric(self) -> float:
    """A metric representing the current performance."""
        pass
```
``Trainer`` is an example of ``Searchable``, and can be used in a ``Search``.

# Features

* **Native support for hyperparameter search**: using search tags (see ``!g`` in the example) users can define multi variant pipelines.
* **Remote and distributed experiments**: users can submit ``Experiments`` to ``Clusters`` which will execute in a distributed way. Full ``AWS`` integration is supported.
* **Visualize all your metrics and meaningful data using Tensorboard**: log scalars, histograms, images, hparams and much more.
* **Add custom code and objects to your pipelines**: extend flambé functionality using our easy-to-use *extensions* mechanism.
* **Modularity with hierarchical serialization**: save different components from pipelines and load them safely anywhere.

# Next Steps

Full documentation, tutorials and much more in https://flambe.ai

# Cite

You can cite us with:

@inproceedings{wohlwend-etal-2019-flambe,
    title = "{F}lamb{\'e}: A Customizable Framework for Machine Learning Experiments",
    author = "Wohlwend, Jeremy  and
      Matthews, Nicholas  and
      Itzcovich, Ivan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-3029",
    doi = "10.18653/v1/P19-3029",
    pages = "181--188"
}

# Contact

You can reach us at flambe@asapp.com
