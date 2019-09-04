.. _starting-quickstart_label:

==========
Quickstart
==========

Flambé runs processes that are described using `YAML <https://en.wikipedia.org/wiki/YAML>`_ files.
When executing, Flambé will automatically convert these processes into Python objects and it will start
executing them based on their behavior.

One of the processes that Flambé is able to run is an :class:`~flambe.experiment.Experiment`:

.. code-block:: yaml
    :caption: simple-exp.yaml

    !Experiment

    name: sst

    pipeline:

      # stage 0 - Load the dataset object SSTDataset and run preprocessing
      dataset: !SSTDataset
        transform:
          text: !TextField  # Another class that helps preprocess the data
          label: !LabelField

This :class:`~flambe.experiment.Experiment` just loads the
`Stanford Sentiment Treebank <https://nlp.stanford.edu/sentiment/treebank.html>`_
dataset which we will use later.

.. important::
    Note that all the keywords following ``!`` are just Python classes
    (``Experiment``, ``SSTDataset``, ``TextField``, ``LabelField``)
    whose keyword parameters are passed to the ``__init__`` method.

.. _starting-executing_label:

Executing Flambé
----------------

Flambé can execute the previously defined ``Experiment`` by running:

.. code-block:: bash

    flambe simple-exp.yaml

Because of the way ``Experiments`` work, flambé will start executing the ``pipeline``
sequentially. Once done, you should see the generated artifacts in ``flambe-output/output__sst/``.
Obviously, these artifacts are useless at this point. Let's add a Text Classifier model
and train it with this same dataset:

.. seealso::
    For a better understanding of :class:`~flambe.experiment.Experiment` read the
    :ref:`understanding-experiments_label` section.


A Simple Experiment
-------------------

Lets add a second stage to the ``pipeline`` to declare a text classifier.
We can use Flambé's :class:`~flambe.nlp.classification.TextClassifier`:

.. code-block:: yaml

    !Experiment

    name: sst
    pipeline:

      # stage 0 - Load the dataset object SSTDataset and run preprocessing
      [...]  # Same as before


      # stage 1 - Define the model
      model: !TextClassifier
        embedder: !Embedder
          embedding: !torch.Embedding
            num_embeddings: !@ dataset.text.vocab_size
            embedding_dim: 300
          encoder: !PooledRNNEncoder
            input_size: 300
            rnn_type: lstm
            n_layers: !g [2, 3, 4]
            hidden_size: 256
        output_layer: !SoftmaxLayer
          input_size: !@ model[embedder][encoder].rnn.hidden_size
          output_size: !@ dataset.label.vocab_size

By using ``!@`` you can link to attributes of previously defined objects. Note that we take
``num_embeddings`` value from the dataset's vocabulary size that it is stored in its ``text`` attribute.
These are called ``Links`` (read more about them in :ref:`understanding-links_label`).

Links always start from the top-level stage in the pipeline, and can even be self-referential,
as the second link references the model definition it is a part of:

.. code-block:: yaml

      input_size: !@ model[embedder][encoder].rnn.hidden_size

Note that the path starts from ``model`` and the brackets access the
embedder and then the encoder in the config file. You can then use dot
notation to access the runtime instance attributes of the target object, the
encoder in this example.

Always refer to the documentation of the object you're linking to in order to understand
what attributes it actually has when the link will be resolved.

.. important::
    You can only link to non-parent objects above the position of the link in the
    config file, because later objects, and parents of the link, will not be initialized
    at the time the link is resolved.


.. important::
    **Flambé supports native hyperparameter search!**

    .. code-block:: yaml

        n_layers: !g [2, 3, 4]

    Above we define 3 variants of the model, each containing different
    amount of ``n_layers`` in the ``encoder``.

Now that we have the dataset and the model, we can add a training process. Flambé provides
a powerful and flexible implementation called :class:`~flambe.learn.Trainer`:

.. code-block:: yaml

    !Experiment

    name: sst
    pipeline:

      # stage 0 - Load the dataset object SSTDataset and run preprocessing
      [...]  # Same as before


      # stage 1 - Define the model
      [...]  # Same as before

      # stage 2 - train the model on the dataset
      train: !Trainer
        dataset: !@ dataset
        train_sampler: !BaseSampler
          batch_size: 64
        val_sampler: !BaseSampler
        model: !@ model
        loss_fn: !torch.NLLLoss  # Use existing PyTorch negative log likelihood
        metric_fn: !Accuracy  # Used for validation set evaluation
        optimizer: !torch.Adam
          params: !@ train[model].trainable_params
        max_steps: 20
        iter_per_step: 50

.. tip::
    Flambé provides full integration with Pytorch object by using
    ``torch`` prefix. In this example, objects like ``NLLLoss`` and
    ``Adam`` are directly used in the configuration file!

.. tip::
  Additionally we setup some ``Tune`` classes for use with hyperparameter search and scheduling.
  They can be accessed via ``!ray.ClassName`` tags. More on hyperparameter search and
  scheduling in :ref:`understanding-experiments_label`.


Monitoring the Experiment
-------------------------

Flambé provides a powerful UI called the **Report Site** to monitor progress in real time.
It has full integration with `Tensorboard <https://www.tensorflow.org/guide/summaries_and_tensorboard>`_.

When executing the experiment (see :ref:`starting-executing_label`), flambé will show instructions
on how to launch the Report Site.

.. seealso::
  Read more about monitoring in :ref:`understanding-report-site_label` section.

Artifacts
---------

By default, artifacts will be located in ``flambe-ouput/`` (relative the the current work directory). This behaviour
can be overriden by providing a ``save_path`` parameter to the ``Experiment``.

::

    flambe-output/output__sst
    ├── dataset
    │   └── 0_2019-07-23_XXXXXX
    │       └── checkpoint
    │           └── checkpoint.flambe
    │               ├── label
    │               └── text
    ├── model
    │   ├── n_layers=2_2019-07-23_XXXXXX
    │   │    └── checkpoint
    │   │        └── checkpoint.flambe
    │   │            ├── embedder
    │   │            │   ├── embedding
    │   │            │   └── encoder
    │   │            └── output_layer
    │   ├── n_layers=3_2019-07-23_XXXXXX
    │   │    └── ...
    │   └── n_layers=4_2019-07-23_XXXXXX
    │       └── ...
    └── trainer
        ├── n_layers=2_2019-07-23_XXXXXX
        │    └── checkpoint
        │        └── checkpoint.flambe
        │            ├── model
        │            │   ├── embedder
        │            │   │   └── ...
        │            │   └── output_layer
        │            └── dataset
        │                └── ...
        ├── n_layers=3_2019-07-23_XXXXXX
        │    └── ...
        └── n_layers=4_2019-07-23_XXXXXX
             └── ...

**Note that the output is 100% hierarchical**. This means that each component is isolated and reusable by itself.

:meth:`~flambe.compile.serialization.load` is a powerful utility to load previously saved objects.

.. code-block:: python
    :linenos:

    import flambe

    path = "flambe-output/output__sst/train/n_layers=4_.../.../model/embedder/encoder/"
    encoder = flambe.load(path)

.. important::
    The output folder also reflects the variants that were speficied
    in the config file. There is one folder for each variant in ``model``
    and in ``trainer``. **The** ``trainer`` **inherits the variants from the previous
    components, in this case the** ``model``. For more information on variant inheritance,
    go to :ref:`understanding-search-options_label`.

Recap
-----

You should be familiar now with the following concepts

* ``Experiments`` can be represented in a YAML format where a ``pipeline`` can be specified,
  containing different components that will be executed sequentially.
* Objects are referenced using ``!`` + the class name. Flambé will compile this structure into a Python
  object.
* Flambé supports natively searching over hyperparameters with tags like ``!g`` (to perform Grid
  Search).
* References between components are done using ``!@`` links.
* The Report Site can be used to monitor the ``Experiment`` execution, with full integration with
  Tensorboard.

Try it yourself!
----------------

Here is the full config we used in this tutorial:

.. code-block:: yaml
    :linenos:
    :caption: simple-exp.yaml

    !Experiment

    name: sst
    pipeline:

      # stage 0 - Load the dataset object SSTDataset and run preprocessing
      dataset: !SSTDataset
        transform:
          text: !TextField  # Another class that helps preprocess the data
          label: !LabelField


      # stage 1 - Define the model
      model: !TextClassifier
        embedder: !Embedder
          embedding: !torch.Embedding
            num_embeddings: !@ dataset.text.vocab_size
            embedding_dim: 300
          encoder: !PooledRNNEncoder
            input_size: 300
            rnn_type: lstm
            n_layers: !g [2, 3, 4]
            hidden_size: 256
        output_layer: !SoftmaxLayer
          input_size: !@ model[embedder][encoder].rnn.hidden_size
          output_size: !@ dataset.label.vocab_size

      # stage 2 - train the model on the dataset
      train: !Trainer
        dataset: !@ dataset
        train_sampler: !BaseSampler
          batch_size: 64
        val_sampler: !BaseSampler
        model: !@ model
        loss_fn: !torch.NLLLoss  # Use existing PyTorch negative log likelihood
        metric_fn: !Accuracy  # Used for validation set evaluation
        optimizer: !torch.Adam
          params: !@ train[model].trainable_params
        max_steps: 20
        iter_per_step: 50

We encourage you to execute the experiment and to start getting familiar with the
artifacts and the report site.

Next Steps
----------

* :ref:`understanding-component_label`: ``SSTDataset``, ``Trainer`` and ``TextClassifier`` are examples of :class:`~flambe.compile.Component`.
  These objects are the core of the experiment's ``pipeline``.
* :ref:`understanding-runnables_label`: flambé supports running multiple processes, not just ``Experiments``.
  These objects must implement :class:`~flambe.runnable.Runnable`.
* :ref:`understanding-clusters_label`: learn how to create clusters and run
  remote experiments.
* :ref:`understanding-extensions_label`: flambé provides a simple and easy mechanism to declare custom
  :class:`~flambe.runnable.Runnable` and :class:`~flambe.compile.Component`.
* :ref:`understanding-experiments-scheduling_label`: besides grid search, you might also want to try out more sophisticated
  hyperparameter search algorithms and resource allocation strategies like Hyperband.
