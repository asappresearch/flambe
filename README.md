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
| [Script](#script) | An entry-point for users who wish to keep their code unchanged, but leverage Flambé's cluster management and distributed hyperparameter search.|
| [PytorchTask](#trainer) | Train / Evaluate a single model on a given task. Offers an interface to automate the boilerplate code usually found in PyTorch code, such as multi-gpu handling, fp16 training, and training loops. |


### Execute a task

Flambé executes tasks by representing them through a YAML configuration:

<table>
<tr style="font-weight:bold;">
  <td>Python Code</td>
  <td>YAML Config</td>
  </tr>
<tr>
<td valign="top">
<pre lang="python">
 
    script = flambe.Script(
      path='path/to/script/',
      args={
         'arg1' = 0.5,
         'arg2' = 2,
      }
    )
</pre>
</td>
<td valign="top">
<pre lang="yaml">

    !Script

    path: path/to/script
    args:
      arg1: 0.5
      arg2: 2
</pre>
</td>
</tr>
<tr>
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

#### Example: Script

Run a hyperparameter search over a Python script:

<table>
<tr style="font-weight:bold;">
  <td>Python Code</td>
  <td>YAML Config</td>
  </tr>
<tr>
<td valign="top">
<pre lang="python">

    import flambe as fl

    script = fl.learn.Script.schema(
      path='path/to/script/',
      output_arg='output-path'
      args={
         'arg1' = fl.uniform(0, 1)
         'arg2' = fl.choice([1, 2, 3])
      }
    )
    
    algorithm = fl.RandomSearch(trial_budget=3)
    search = fl.Search(script, algorithm)
    search.run()
</pre>
</td>
<td valign="top">
<pre lang="yaml">
    
    !Search
    
    searchable: !Script
      path: path/to/script
      output_arg: output-path
      args:
        arg1: !~u [0, 1]
        arg2: !~c [1, 2, 3]
    algorithm: !RandomSearch
      trial_budget=3
</pre>
</td>
</tr>
</table>

#### Example: PyTorchTask

Run a hyperpameter search over a ``PyTorchTask``: 

<table>
<tr style="font-weight:bold;">
  <td>Python Code</td>
  <td>YAML Config</td>
  </tr>
<tr>
<td valign="top">
<pre lang="python">

    import flambe as fl
 
    with flambe.as_schemas():
      dataset = fl.nlp.SSTDataset()
      model = fl.nlp.TextClassifier(
          n_layers=fl.choice([1, 2, 3])  
      )
      trainer = fl.learn.Trainer(
          dataset=dataset,
          model=model
      )
 
    algorithm = fl.RandomSearch(max_steps=10, trial_budget=2)
    search = Search(searchable=trainer, algorithm=algorithm)
    search.run()
</pre>
</td>
<td valign="top">
<pre lang="yaml">

    !Search
  
    searchable: !Trainer
       dataset: !SSTDataset
       model: !TextClassifier
          n_layers: !~c [1, 2, 3]
    algorithm: !RandomSearch
      max_steps: 10
      trial_budget: 2
</pre>
</td>
</tr>
</table>

**Note**: the method ``schema`` enables passing distributions as input arguments, which is automatic in YAML.  

### Build a pipeline.

For more information see: [here](http://flambe.ai/).

Construct a computational graph with different training stages:

<table>
<tr style="font-weight:bold;">
  <td>Python Code</td>
  <td>YAML Config</td>
  </tr>
<tr>
<td valign="top">
<pre lang="python">

    import flambe as fl
 
    with flambe.as_schemas():
      dataset = fl.nlp.SSTDataset()
      model = fl.nlp.TextClassifier(
          n_layers=fl.choice([1, 2, 3])  
      )
      pretrain = fl.learn.Trainer(
          dataset=dataset,
          model=model
      )
      
      dataset = fl.nlp.SSTDataset()
      finetune = fl.learn.Trainer(
          dataset=dataset,
          model=trainer.model,
          dropout: fl.uniform(0, 1)
      )
 
    algorithm1 = fl.RandomSearch(max_steps=10, trial_budget=2)
    algorithm2 = fl.RandomSearch(max_steps=10, trial_budget=2)
    
    pipeline = dict(pretrain=pretrain, finetune=finetune)
    algorithm = dict(pretrain=algorithm1, finetune=algorithm2)
    
    exp = Experiment(pipeline, algorithm)
    exp.run()
 
</pre>
</td>
<td valign="top">
<pre lang="yaml">

    !Pipeline
  
    pipeline:
     pretraining: !Trainer
       dataset: !SSTDataset
       model: !TextClassifier
          n_layers: !~c [1, 2, 3]
     finetuning: !Trainer
       dataset: !SSTDataset
       dropout: !~u [0, 1]
       model: !@ pretraining[model]
   
    algorithm:
      pretraining:!RandomSearch
        max_steps: 10
        trial_budget: 2
      finetuning:!RandomSearch
        max_steps: 10
        trial_budget: 2
</pre>
</td>
</tr>
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
