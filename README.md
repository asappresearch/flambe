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

* **Run hyperparameter searches** over arbitrary Python objects or scripts.
* **Constuct DAGs**, by searching over hyperparameters and reducing to the
best variants at any of the nodes.
* Distribute tasks **remotely** and **in parallel** over a cluster, with full AWS,
GCP, and Kubernetes integrations.
* Easily **share** experiment configurations, results, and checkpoints with others.
* Automate the **boilerplate code** in training models with [PyTorch.](https://pytorch.org)

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

Flambé executes ``Runnables``, which are simply Python objects that implement the method ``run``.  
Flambé provides the following set of ``Runnables``, but you can easily create your own:

| Runnable | Description |
| -------|------|
| [Script](#script) | An entry-point for users who wish to keep their code unchanged, but leverage Flambé's cluster management and distributed hyperparameter search tools.|
| [Trainer](#trainer) | Train / Evaluate a single model on a given task. Offers an interface to automate the boilerplate code usually found in PyTorch scripts, such as multi-gpu handling, fp16 training, and training loops. |
| [Search](#search) | Run a hyperparameter search over python objects and scripts. |
| [Experiment](#experiment) | Build a computational DAG, with the possibility of running a hyperparameter search at each node, and reduce to the best variants |

``Runnables`` can be used in regular python scripts, or executed through YAML configurations with the command:

```bash
flambe run [CONFIG]
```

or to submit to a cluster:

```bash
flambe submit [CONFIG] --cluster ~/.flambe/cluster.yaml
```

In the following examples, each code snippet is shown alongside its corresponding YAML configuration.

### Script

<table>
<tr style="font-weight:bold;">
  <td>Python Code</td>
  <td>YAML Config</td>
  </tr>
<tr>
<td valign="top">
<pre lang="python">
    
    import flambe as fl

    script = fl.Script(
      path='path/to/script/',
      output_arg='output-path'
      args={
         'arg1' = 0.5
      }
    )

    script.run()
</pre>
</td>
<td valign="top">
<pre lang="yaml">

    !Script

    path: path/to/script
    output_arg: output-path
    args:
      arg1: 0.5
      arg2: 2
</pre>
</td>
</tr>
</table>

To run a hyperparameter search over your script, see [here](#search). 

### Trainer

<table>
<tr style="font-weight:bold;">
  <td>Python Code</td>
  <td>YAML Config</td>
  </tr>
<tr>
<td valign="top">
<pre lang="python">

    import flambe as fl
    
    dataset = fl.nlp.SSTDataset()
    model = fl.nlp.TextClassifier(
        n_layers=2
    )
    trainer = fl.learn.Trainer(
        dataset=dataset,
        model=model
    )
 
    trainer.run()
</pre>
</td>
<td valign="top">
<pre lang="yaml">

    !Trainer
    
    dataset: !SSTDataset
    model: !TextClassifier
       n_layers: 2
</pre>
</td>
</tr>
</table>

To run a hyperparameter search over a trainer, see [here](#search).    

### Search

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

**Note**: the method ``schema`` enables passing distributions as input arguments, which is automatic in YAML.  

Run a hyperpameter search over a ``Trainer``: 

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


### Experiment

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

    !Experiment
  
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
