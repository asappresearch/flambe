.. raw:: html

    <p align="center">
       <img src="imgs/Flambe_Logo_CMYK_FullColor.png" width="60%" align="middle">
    </p>

|

------------

|

.. image:: https://img.shields.io/circleci/build/github/asappresearch/flambe
    :target: https://circleci.com/gh/asappresearch/flambe

.. image:: https://readthedocs.org/projects/flambe/badge/?version=latest
    :target: https://flambe.ai/en/latest/?badge=latest
    :alt: Documentation Status

|

Welcome to Flambé, a `PyTorch <https://pytorch.org/>`_-based library that allows users to:

* Run complex experiments with **multiple training and processing stages**
* **Search over hyperparameters**, and select the best trials
* Run experiments **remotely** over many workers, including full AWS integration
* Easily share experiment configurations, results, and model weights with others

Flambe offers an end to end experience by connecting the dots between a curated set
of frameworks. In doing so, Flambe is focused on creating the best user experience possible.

Installation
------------

**From** ``PIP``:

.. code-block:: bash

    pip install flambe

**From source**:

.. code-block:: bash

    git clone git@github.com:asappresearch/flambe.git
    cd flambe
    pip install .


Getting started
---------------

Training a model
################

.. code-block:: python
    import flambe as fl

    model = fl.nlp.TextClassifier(n_layers=2)
    trainer = fl.learn.Trainer(model)

    # Execute training
    trainer.run()


Run a hyperparameter search
###########################


.. code-block:: python
    import flambe as fl

    model = fl.schema(
      fl.nlp.TextClassifier,
      n_layers=fl.choice([1, 2, 3])
    )
    trainer = fl.schema(
      fl.learn.Trainer,
      model=model
    )

    # Run a hyperparameter search
    algorithm = fl.RandomSearch(max_steps=10, trial_budget=2)
    search = Search(trainer, algorithm)
    search.run()

A schema is a representation of an object that accepts search options
as arguments to the object's constructor. The schema passed to ``Search``
object must be a wrapper around an object that follows the Searchable
interface below. 

.. code-block:: python

    class Searchable:

      def step() -> bool:
        """Indicate whether execution is complete."""
      
      def metric -> float:
        """A metric representing the current performance."""

``Trainer`` is an example of an object that implements this interface,
and can therefore be used in a ``Search``.

Features
--------

* **Native support for hyperparameter search**: using search tags (see ``!g`` in the example) users can define multi variant pipelines.
* **Remote and distributed experiments**: users can submit ``Experiments`` to ``Clusters`` which will execute in a distributed way. Full ``AWS`` integration is supported.
* **Visualize all your metrics and meaningful data using Tensorboard**: log scalars, histograms, images, hparams and much more.
* **Add custom code and objects to your pipelines**: extend flambé functionality using our easy-to-use *extensions* mechanism.
* **Modularity with hierarchical serialization**: save different components from pipelines and load them safely anywhere.

Next Steps
-----------

Full documentation, tutorials and much more in https://flambe.ai

Contact
-------
You can reach us at flambe@asapp.com
