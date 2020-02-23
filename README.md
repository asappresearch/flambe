<p align="center">
  <img src="imgs/Flambe_Logo_CMYK_FullColor.png" width="500" align="middle">
</p>

------------

[![Fast Tests Status](https://github.com/asappresearch/flambe/workflows/Run%20fast%20tests/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Slow Tests Status](https://github.com/asappresearch/flambe/workflows/Run%20slow%20tests/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Documentation Status](https://readthedocs.org/projects/flambe/badge/?version=latest)](https://flambe.ai/en/latest/?badge=latest)
[![PyPI Version](https://badge.fury.io/py/flambe.svg)](https://badge.fury.io/py/flambe)

Flambé is a Python framework built to accelerate the development of machine learning research.

With Flambé you can:

* **Run hyperparameter searches** over any Python code.
* **Constuct pipelines**, searching over hyperparameters and reducing to the
best variants at any step.
* Automate the **boilerplate code** in training models with [PyTorch.](https://pytorch.org)
* Distribute jobs **remotely** and **in parallel** over a cluster, with full AWS,
GCP, and Kubernetes integrations.
* Easily **share** experiment configurations, results, and checkpoints with others.


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

Create a Flambé ``Task``, by writting two simple methods:

```python

class Task:
  
  # REQUIRED
  def run(self) -> bool:
    """Execute a computational step, returns True until done."""
    ...
  
  # OPTIONAL
  def metric(self) -> float:
    """Get the current performance on the task to compare with other runs."""
    ...
  
```

Flambé provides the following set of tasks, but you can easily create your own:

| Task | Description |
| -------|------------|
| [Script](http://flambe.ai/) | An entry-point for users who wish to keep their code unchanged, but leverage Flambé's cluster management and distributed hyperparameter search.|
| [PytorchTask](http://flambe.ai/) | Train / Evaluate a single model on a given task. Offers an interface to automate the boilerplate code usually found in PyTorch code, such as multi-gpu handling, fp16 training, and training loops. |

Flambé also provides a set of **meta-tasks**, which are tasks that operate over other tasks:

| Task | Description |
| -------|------------|
| [Search](http://flambe.ai/) | Run a hyperparameter search over another task by replacing any arguments by a distribution. |
| [Pipeline](http://flambe.ai/) | Build a pipeline of tasks, run a hyperparameter search and reduce to the best variants and any step.

### Execute a task

Flambé executes tasks by representing them through a YAML configuration file:

<table>
<tr style="font-weight:bold;">
  <td>Python Code <img width=310/></td>
  <td>YAML Config <img width=310/></td>
  </tr>
<tr>
<td valign="top">
<pre lang="python">
 
    script = Script(
      path='flambe/examples/script.py',
      args={
         'dropout' = 0.5,
         'num_layers' = 2,
      }
    )
</pre>
</td>
<td valign="top">
<pre lang="yaml">

    !Script

    path: flambe/examples/script.py
    args:
      dropout: 0.5
      num_layers: 2
</pre>
</td>
</tr>
</table>

Execute a task locally with:

```bash
flambe run [CONFIG]
```

Submit a task to a cluster with:

```bash
flambe submit [CONFIG] -c [CLUSTER]
```

For more information on remote execution, and how to create a cluster see: [here](http://flambe.ai/).

### Run a hyperparameter search

Search over arguments to your ``Task`` and execute with the algorithm of your choice.

<table>
<tr style="font-weight:bold;">
  <td>YAML Config <img width=800/></td>
  </tr>
<tr>
<td valign="top">
<pre lang="yaml">

    !Search

    task: !Script
      path: flambe/examples/script.py
      args:
        dropout: !uniform [0, 1]
        num_layers: !choice [1, 2, 3]

    algorithm: !RandomSearch
      trial_budget=3
</pre>
</td>
</table>

### Build a pipeline.

A Flambé pipeline may contain any of the following:

1. Other Flambé tasks to execute
2. Other Flambé tasks containing search options to search over
3. Other Python objects that will not be executed but help define the pipeline

``Pipelines`` are useful when you have dependencies between tasks, for example in a pretrain then finetune setting:

<table>
<tr style="font-weight:bold;">
  <td>YAML Config <img width=800/></td>
  </tr>
<tr>
<td valign="top">
<pre lang="yaml">

    !Pipeline

    pipeline:
      pretrain: !Script
        path: flambe/examples/pretrain.py
        args:
          dropout: !uniform [0, 1]
          num_layers: !choice [1, 2, 3]
      finetune: !Script
        path: flambe/examples/finetune.py
        args:
          checkpoint: !copy pretrain[path]
          learning_rate: !choice [.001, .0001]

    algorithm:
      pretrain: !RandomSearch
        max_steps: 100
        trial_budget: 5
      finetune: !RandomSearch
        max_steps: 10
        trial_budget: 2

    reduce:
      pretrain: 2
</pre>
</td>
</table>

## Next Steps

Full documentation, tutorials and much more at [flambe.ai](http://flambe.ai/).

## Contributing

See [contributing](https://github.com/asappresearch/flambe/blob/master/CONTRIBUTING.md) and our [style guidelines](https://github.com/asappresearch/flambe/blob/master/STYLE.md).

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
