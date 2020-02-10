<p align="center">
  <img src="imgs/Flambe_Logo_CMYK_FullColor.png" width="500" align="middle">
</p>

------------

[![Fast Tests Status](https://github.com/asappresearch/flambe/workflows/tests-fast/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Slow Tests Status](https://github.com/asappresearch/flambe/workflows/tests-slow/badge.svg)](https://github.com/asappresearch/flambe/actions)
[![Documentation Status](https://readthedocs.org/projects/flambe/badge/?version=latest)](https://flambe.ai/en/latest/?badge=latest)

Flambé is a Python framework built to accelerate the development of machine learning research.
Flambé connects the dots between a curated set of libraries to provide a unified experience.

With Flambé you can:

* **Run hyperparameter searches** over arbitrary Python objects or scripts.
* **Constuct DAGs**, by searching over hyperparameters and reducing to the
best variants at any of the nodes.
* Distribute tasks **remotely** and **in parallel** over a cluster, with full AWS,
GCP, and Kubernetes integrations.
* Easily **share** experiment configurations, results, and model weights with others.
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
| Script | Execute a python script |
| Learner | Train / Evaluate a single model on a given task |
| Search | Run a hyperparameter search |
| Experiment | Build a computational DAG, with with a search at any node |

``Runnables`` can be executed in regular python scripts or through the ``flambe run [CONFIG]`` command.  
The command takes as input a YAML configuration representing the object to execute.

In the following examples, each code snippet is shown alongside its corresponding YAML configuration.


### Sript

``Script`` provides an entry-point for users who wish to keep their code unchanged, and
only leverage Flambé's cluster management and distributed hyperparameter search tools.

<table>
<tr style="font-weight:bold;">
  <td>Code</td>
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
         'arg1' = 1
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
      arg1: 1
  </pre>
</td>
</tr>
</table>

### Learner

In cases where you are starting to build your machine learning project from scratch,
the ``Learner`` offers an interface to reduce the boilerplate code usually found
in PyTorch scripts, such as multi-gpu handling, fp16 training, and training loops.


<table>
<tr style="font-weight:bold;">
  <td>Code</td>
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

In the snippet below, we show how to convert a training routine to a hyperparameter search.
Any python object can be turned into a ``Schema`` which accept distribution as arguments.

<table>
<tr style="font-weight:bold;">
  <td>Code</td>
  <td>YAML Config</td>
  </tr>
<tr>
<td valign="top">
   <pre lang="python">

    import flambe as fl
 
    with flambe.search():
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

Flambé also offers a set of commands 

### Running jobs on a cluster

Flambé provides a simple wrapper over the Ray Autoscaler, which enables
creating clusters of machines, and distributing jobs onto the cluster.
Flambé provides the following set of commands to submit runnable configurations
to the cluster and managing running jobs.

| Command | Description |
| -------|------|
| up | Start or update the cluster. |
| down | Teardown the cluster. |
| submit | Submit a job to the cluster, as a YAML config. |
| ls | List the jobs (i.e tmux sessions) running on the cluster. |
| attach |  Attach to a running job (i.e tmux session) on the cluster. |
| site | Launch a Web UI to monitor the activity on the cluster. |
| kill | ill a job (i.e tmux session) running on the cluster. |
| clean | Clean the artifacts of a job on the cluster.|
| exec | Execute a command on the cluster head node. |
| rsync-up | Upload files to the cluster. |
| rsync-down |  Download files from the cluster. |
          

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
