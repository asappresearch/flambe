<p align="center">
  <img src="imgs/Flambe_Logo_CMYK_FullColor.png" width="500" align="middle">
</p>

------------

[![Fast Tests Status](https://github.com/asappresearch/flambe/workflows/Run%20fast%20tests/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Slow Tests Status](https://github.com/asappresearch/flambe/workflows/Run%20slow%20tests/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Documentation Status](https://readthedocs.org/projects/flambe/badge/?version=latest)](https://flambe.ai/en/latest/?badge=latest)
[![PyPI Version](https://badge.fury.io/py/flambe.svg)](https://badge.fury.io/py/flambe)

Flambé is a Python framework built to **accelerate the machine learning research lifecycle**.

Running a machine learning experiment generally involves the following steps:

1. Write data processing, model training and evaluation code for the task
2. Execute the task on a remote machine or cluster
3. Improve performance by searching across models and hyperparameters
4. Export a final model for inference

Flambé eliminates the boilerplate involved in the first step, and fully automates the others.  
The full documentation can be found here: [flambe.ai](https://flambe.ai).

## Installation

> Note: Flambé currently only supports Python 3.7.x

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

### 1. Create a task

Write a task from scratch by implementing two simple methods:

```python

class MyTask:

  # REQUIRED
  def train(self) -> bool:
    """Train for an epoch, keeps returning True until complete."""
    ...

  # OPTIONAL
  def validate(self) -> float:
    """Get the current performance on the task to compare with other runs."""
    ...

```

or start from one of the provided templates:

| Task | Description |
| -------|------------|
| [Script](http://flambe.ai/) |  Wrapper over any python script that accepts command line arguments. An entry-point for users who wish to keep their code unchanged, but leverage Flambé's cluster management and distributed hyperparameter search.|
| [Lightning](http://flambe.ai/) |  Train / Evaluate a model using Pytorch Lightning. Automates the boilerplate code usually found in PyTorch code, such as multi-gpu handling, fp16 training, and training loops. |
| [RaySGD](http://flambe.ai/) | Train / Evaluate a model using RaySGD. Automate the boilerplate code usually found in PyTorch or Tensorflow code, such as multi-gpu handling, fp16 training, and training loops. |

### 2. Execute a task

Define your task as a YAML configuration:

<table>
<tr style="font-weight:bold;">
  <td>Python Code <img width=310/></td>
  <td>YAML Config <img width=310/></td>
  </tr>
<tr>
<td valign="top">
<pre lang="python">

    import my_module

    task = my_module.MyTask(
       dropout=0.5,
       num_layers=2
    )

    task.run()
</pre>
</td>
<td valign="top">
<pre lang="yaml">

    !my_module.MyTask

     dropout: 0.5
     num_layers: 2
</pre>
</td>
</tr>
</table>

Execute the task locally with:

```bash
flambe run [CONFIG]
```

or remotely with:

```bash
flambe submit [CONFIG] -c [CLUSTER]
```

For more information on remote execution, and how to create a cluster see: [here](http://flambe.ai/).

### 3. Search over models and hyperparameters

#### ``Search``

Use the built-in ``Search`` to run distributed hyperparameter searches over other Flambé tasks.

For instance, you can run a hyperparameter search over any python script with a few lines:

<table>
<tr style="font-weight:bold;">
  <td>YAML Config <img height=0 width=800/></td>
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
      trial_budget: 5
</pre>
</td>
</table>

#### ``Pipeline``

Use the built-in ``Pipeline`` to run multiple searches, and reduce to the best variants at any stage. This is useful when your experiment involves multiple tasks that may depend on each other.

A Flambé pipeline may contain any of the following:

1. Flambé tasks to execute
2. Flambé tasks containing search options to search over (in this case a ``Search`` is automatically created.)
3. Other Python objects that will not be executed but help define the pipeline

<table>
<tr style="font-weight:bold;">
  <td>YAML Config <img height=0 width=800/></td>
</tr>
<tr>
<td valign="top">
<pre lang="yaml">

    !Pipeline

    stages:
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
        trial_budget: 5
      finetune: !RandomSearch
        trial_budget: 2

    reduce:
      pretrain: 2
</pre>
</td>
</table>

### 4. Export a model for inference

Wether you executed your task on its own, in a ``Search`` or in a ``Pipeline``, Flambé
will save artifacts of your experiment. You can then construct your final inference object:

<table>
<tr style="font-weight:bold;">
  <td>YAML Config <img height=0 width=800/></td>
</tr>
<tr>
<td valign="top">
<pre lang="yaml">

    !Export

    object: !MyInferenceModel
      vocabulary: path/to/vocabulary
      model: path/to/model

    format: pickle # onnx, torchscript
    compress: true
    
    dest_type: local # s3, ...
    dest_path: path/to/output
</pre>
</td>
</table>


## Next steps

Full documentation, tutorials and more at [flambe.ai](http://flambe.ai/).

## Contributing

See [contributing](https://github.com/asappresearch/flambe/blob/master/CONTRIBUTING.md) and our [style guidelines](https://github.com/asappresearch/flambe/blob/master/STYLE.md).

## Cite

You can cite us with:

```bash
@inproceedings{wohlwend-etal-2019-flambe,
    title = "{F}lamb{\'e}: A Customizable Framework for Machine Learning Pipelines",
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
