.. _Search:

======
Search
======

One of the core objective of Flambé is to make it easy to run hyperparameter
searches over any python code. In this section we provide more information
on how to operate the ``Search`` object.

Schema
------

In order to construct hyperparameter search, we must build a representation
of our Python object that accepts distributions as input argument. The core
mechanism behind this functionality is the ``Schema``. 



Examples
--------

**Run a hyperparameter search over a Python script**:

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

**Run a hyperpameter search over a ``Trainer``**: 

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