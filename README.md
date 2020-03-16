<p align="center">
  <img src="imgs/Flambe_Logo_CMYK_FullColor.png" width="500" align="middle">
</p>

------------

[![Fast Tests Status](https://github.com/asappresearch/flambe/workflows/Run%20fast%20tests/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Slow Tests Status](https://github.com/asappresearch/flambe/workflows/Run%20slow%20tests/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Documentation Status](https://readthedocs.org/projects/flambe/badge/?version=latest)](https://flambe.ai/en/latest/?badge=latest)
[![PyPI Version](https://badge.fury.io/py/flambe.svg)](https://badge.fury.io/py/flambe)

Flambé is a Python framework built to **accelerate the machine learning research lifecycle**.

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Cite](#cite)
- [Contact](#contact)

The full documentation can be found here: [flambe.ai](https://flambe.ai).

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

> Note: Flambé currently only supports Python 3.7.x

## Getting started

Running a machine learning experiment generally involves the following steps:

1. Write data processing, model training and evaluation code for the task
2. Execute the task on a remote machine or cluster
3. Improve performance by searching across models and hyperparameters
4. Export a final model for inference

Flambé eliminates much of the boilerplate involved in the first step, and fully automates the others.  

### 1. Create a task

Write a task from scratch by implementing two simple methods:

```python
class MyTask:

  # REQUIRED
  def train(self, n_epochs: float = 1.0):
    """Run training for the given number of epochs."""
    ...

  # OPTIONAL
  def validate(self) -> float:
    """Run validation and return a metric to use to compare with other runs."""
    ...
```

or start from one of the provided templates:

| Task | Description |
| -------|------------|
| [Script](http://flambe.ai/) |  Wrapper over any python script that accepts command line arguments. An entry-point for users who wish to keep their code unchanged, but leverage Flambé's cluster management and distributed hyperparameter search.|
| [Lightning](http://flambe.ai/) |  Train a model using Pytorch Lightning. Automates the boilerplate code usually found in PyTorch code, such as multi-gpu handling, fp16 training, and training loops. |
| [RaySGD](http://flambe.ai/) | Train a model using RaySGD. Automate the boilerplate code usually found in PyTorch or Tensorflow code, such as multi-gpu handling, fp16 training, and training loops. |
| [Torch](http://flambe.ai/) | Flambé wrapper on top of RaySGD. Designed to use the Flambé provided datasets, featurizers, metrics and pytorch neural network components. |


### 2. Execute a task

Use your task in a script:


```python
import my_module

task = my_module.MyTask(
   dropout=0.5,
   num_layers=2
)

for _ in range(10):
  task.train()
  task.validate()
```

Execute the script locally with:

```bash
flambe run script.py
```

or on a remote cluster with:

```bash
flambe submit script.py -c [CLUSTER-NAME]
```

For more information on remote execution, and how to create a cluster see: [here](http://flambe.ai/).

### 3. Search over models and hyperparameters

#### ``Search``

Use the built-in ``Search`` to run distributed hyperparameter searches over a task:

```python
import flambe
import my_module

# Define your object as a schema, to pass in distributions
task = flambe.schema(my_module.MyTask)(
   dropout=flambe.uniform(0, 1),
   num_layers=flambe.choice(2, 3, 4)
)

# Choose an algorithm and execute the search
algorithm = flambe.RandomSearch(trial_budget=5)
search = Search(task, algorithm)
search.run()
```

Flambé also provides a ``Script`` task to run a hyperparameter search over any command line script:

```python
import flambe

# Define your object as a schema, to pass in distributions
task = flambe.schema(flambe.Script)(
   path='path/to/script.py',
   output_arg='--output-path',  # used to set the output path for the different trials
   kwargs={
     '--dropout': flambe.random(0, 1),
     '--n_layers': flambe.choice([2, 3, 4])
   }  
)

# Choose an algorithm and execute the search
algorithm = flambe.RandomSearch(trial_budget=5)
search = Search(task, algorithm)
search.run()
```

For more information on the ``Script`` feature, see: [here](http://flambe.ai/).
For more information on hyperparameter searches, see: [here](http://flambe.ai/).  


#### ``Pipeline``

Use the built-in ``Pipeline`` to run multiple searches, and reduce to the best variants at any stage. This is useful when your experiment involves multiple tasks that may depend on each other.

A Flambé pipeline may contain any of the following:

1. Tasks to execute
2. Tasks containing search options to search over (in this case a ``Search`` is automatically created.)
3. Other Python objects that will not be executed but help define the pipeline

Note that all objects must be passed as schemas when using ``Pipeline``.

```python
import flambe
import my_module

task1 = flambe.schema(my_module.FirstTask)(
   dropout=flambe.uniform(0, 1),
   num_layers=flambe.choice(2, 3, 4)
)

task2 = flambe.schema(my_module.SecondTask)(
   pretrained=task1.copy('model.state_dict()'),  # Link to attributes of the previous task
   learning_rate=flambe.uniform(0, 1)
)

alg1 = flambe.RandomSearch(trial_budget=5)
alg2 = flambe.Hyperband(step_budget=100)

pipeline = Pipeline(
  stages = {
    'pretrain': task1,
    'finetune': task2
  },
  reduce = {
    'pretrain': 2,  # Only run finetuning for the best 2
  }
)

pipeline.run()
```

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
