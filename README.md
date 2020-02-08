<p align="center">
  <img src="imgs/Flambe_Logo_CMYK_FullColor.png" width="500" align="middle">
</p>

------------

[![Fast Tests Status](https://github.com/asappresearch/flambe/workflows/tests-fast/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Slow Tests Status](https://github.com/asappresearch/flambe/workflows/tests-slow/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Documentation Status](https://readthedocs.org/projects/flambe/badge/?version=latest)](https://flambe.ai/en/latest/?badge=latest)


Welcome to Flambé, a [PyTorch](https://pytorch.org/)-based library that abstracts away
the boilerplate code tradtionally involved in machine learning research. Flambé does not reinvent
the wheel, but instead connects the dots between a curated set of frameworks. With Flambé you can:

* Automate the boilerplate code in training models with PyTorch
* **Run hyperparameter searches** over arbitrary Python objects
* Constuct experiment DAGs, which include searching over hyperparameters and reducing to the
best variants at each node in the DAG.
* Execute tasks **remotely** and **in parallel** over many workers, including full AWS, GCP, and Kubernetes integration
* Easily share experiment configurations, results, and model weights with others


## Installation

**From** ``PIP``:

```bash
pip install flambe
```

**From source**:

```bash
git clone git@github.com:asappresearch/flambe.git
pip install ./flambe
```

## Getting started

There are a few core objects that offer an entrypoint to using Flambé:

| Object | Role |
| -------|------|
| Trainer | Train a single model on a given task |
| Search | Run a hyperparameter search |
| Experiment | Construct a DAG, with a hyperameter search at each node |

In the snippet below, we show how to convert a training routine to a hyperparameter search:

<table>
<tr style="font-weight:bold;">
  <td>Train a model</td>
  <td>Run a hyperparameter search</td>
  </tr>
<tr>
<td valign="top">
   <pre lang="python">

    import flambe as fl
    
    # Define objects
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
   </pre>
</td>
<td valign="top">
  <pre lang="python">

    import flambe as fl
    
    # Define objects as schemas
    dataset = fl.nlp.SSTDataset.schema()
    model = fl.nlp.TextClassifier.schema(
        n_layers=fl.choice([1, 2, 3])  
    )
    trainer = fl.learn.Trainer.schema(
        dataset=dataset,
        model=model
    )

    # Run a hyperparameter search
    algorithm = fl.RandomSearch(max_steps=10, trial_budget=2)
    search = Search(trainer, algorithm)
    search.run()
  </pre>
</td>
</tr>
</table>

**Note**: ``Search`` expects the schema of an object that implements the 
``Searchable`` interface:

```python
class Searchable:

    def step(self) -> bool:
    """Indicate whether execution is complete."""
        pass
    
    def metric(self) -> float:
    """A metric representing the current performance."""
        pass
```
For instance, ``Trainer`` is an example of ``Searchable``, and can therefore be used in a ``Search``.

## Next Steps

Full documentation, tutorials and much more in https://flambe.ai

## Cite

You can cite us with:

```bash
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
```

## Contact

You can reach us at flambe@asapp.com
