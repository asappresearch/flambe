.. _understanding-component_label:

==========
Components
==========

The most important class in Flambé is :class:`~flambe.compile.Component` which implements
loading from YAML (using ``!ClassName`` notation) and saving state.

.. _understanding-component-yaml_label:

Loading and Dumping from YAML
-----------------------------

A :class:`~flambe.compile.Component` can be created from a YAML config representation,
as seen the :ref:`starting-quickstart_label` example.

Lets take the previously used :class:`~flambe.nlp.classification.TextClassifier` component:

.. code-block:: yaml
    :caption: model.yaml

      !TextClassifier
      embedder: !Embedder
        embedding: !torch.Embedding
          num_embeddings: 200
          embedding_dim: 300
        encoder: !PooledRNNEncoder
          input_size: 300
          rnn_type: lstm
          n_layers: 3
          hidden_size: 256
      output_layer: !SoftmaxLayer
        input_size: 256
        output_size: 10

Loading and dumping objects can be done using ``flambe.compile.yaml`` module.

.. code-block:: python
    :linenos:
    
    from flambe.compile import yaml

    # Loading from YAML into a Schema
    text_classifier_schema = yaml.load(open("model.yaml"))
    text_classifier = text_classifier_schema()  # Compile the Schema

    # Dumping object
    yaml.dump(text_classifier, open("new_model.yaml", "w"))

.. important::
  ``Components`` compile to an intermediate state called :class:`~flambe.compile.Schema` when calling
  ``yaml.load()``. This partial representation can be compiled into the final 
  object by calling ``obj()`` (ie executing ``__call__``), as shown in the example above. For more information
  about this, go to :ref:`understanding-component-delayed-init_label`.

.. seealso:: For more examples of the YAML representation of an object look at :ref:`understanding-configuration_label`

.. _understanding-component-state_label:

Saving and Loading State
------------------------

While YAML represents the "architecture" or how to create an instance of some class,
it does not capture the state. For state, ``Components`` rely on a recursive :meth:`~flambe.compile.Component.get_state`
and :meth:`~flambe.compile.Component.load_state` methods that work similarly to PyTorch's
``nn.Module.state_dict`` and ``nn.Module.load_state_dict``:

.. code-block:: python
    :linenos:

    from flambe.compile import yaml

    # Loading from YAML into a Schema
    text_classifier_schema = yaml.load(open("model.yaml"))
    text_classifier = text_classifier_schema()  # Compile the Schema
    
    state = text_classifier.get_state()

    from flambe.nlp.classification import TextClassifier

    another_text_classifier = TextClassifier(...)
    another_text_classifier.load_state(state)


**Semantic Versioning**

In order to identify and describe changes in class definitions, flambé supports
opt-in semantic class versioning. (If you're not familiar with semantic versioning see `this link <https://semver.org/>`_).

Each class has a class property ``_flambe_version`` to prevent conflics when loading
previously saved states.
Initially, all versions are set to ``0.0.0``, indicating that class versioning should
not be used. Once you increment the version, Flambé will then start comparing
the saved class version with the version on the class at load-time.

.. seealso::
    See :ref:`understanding-experiments-custom-state_label` for more information about
    :meth:`~flambe.compile.Component.get_state` and :meth:`~flambe.compile.Component.load_state`.


.. _understanding-component-delayed-init_label:

Delayed Initialization
----------------------

When you load ``Components`` from YAML they are not initialized into objects immediately.
Instead, they are precompiled into a :class:`~flambe.compile.Schema` that you can think
of as a blueprint for how to create the object later.
This mechanism allows ``Components`` to use links and grid search options.

If you load a schema directly from YAML you can compile it into an instance
by calling the schema:

.. code-block:: python
    :linenos:

    from flambe.compile import yaml

    schema = yaml.load('path/to/file.yaml')
    obj = schema()


.. _understanding-component-existing_label:

Core Components
---------------

:class:`~flambe.dataset.Dataset`
    This object holds the training, validation and test data. Its only requirement is to have the three properties: ``train``, ``dev``
    and ``test``, each pointing to a list of examples. For convenience we provide a ``TabularDataset`` implementation of the interface,
    which can load any ``csv`` or ``tsv`` type format.

    .. code-block:: python
        :linenos:

        from flambe.dataset import TabularDataset
        import numpy as np

        # Random dataset
        train = np.random.random((2, 100))
        val = np.random.random((2, 10))
        test = np.random.random((2, 10))

        dataset = TabularDataset(train, val, test)

:class:`~flambe.field.Field`
    A field takes raw examples and produces a ``torch.Tensor`` (or tuple of ``torch.Tensor``).
    We provide useful fields such as ``TextField``, or ``LabelField``
    which perform tokenization and numericalization.

    .. code-block:: python
        :linenos:

        from flambe.field import TextField
        from flambe.tokenizer import WordTokenizer

        import numpy as np

        # Random dataset
        data = np.array(['Flambe is awesome', 'This framework rocks!'])
        text_field = TextField(WordTokenizer())

        # Setup the entire dataset to build vocab.
        text_field.setup(data)
        text_field.vocab_size  # Returns to 9

        text_field.process("Flambe rocks")  # Returns tensor([6, 1])

:class:`~flambe.sampler.Sampler`
    A sampler produces batches of data, as an interator. We provide a simple ``BaseSampler`` implementation, which takes a dataset as input, as well
    as the batch size, and produces batches of data. Each batch is a tuple of tensors, padded to the maximum length along each dimension.

    .. code-block:: python
        :linenos:

        from flambe.sampler import BaseSampler
        from flambe.dataset import TabularDataset
        import numpy as np

        dataset = TabularDataset(np.random.random((2, 10)))

        sampler = BaseSampler(batch_size=4)
        for batch in sampler.sample(dataset):
            # Do something with batch

:class:`~flambe.nn.Module`
    This object is the main model component interface. It must implement the ``forward`` method as PyTorch's ``nn.Module`` requires.

    We also provide additional machine learning components in the ``nn`` submodule, such as ``Encoder``
    with many different implementations of these interfaces.

:class:`~flambe.learn.Trainer`
    A :class:`~flambe.learn.Trainer` takes as input the training and dev samplers, as well as a model and an optimizer.
    By default, the object keeps track of the last and best models, and each call to run is considered to be an arbitrary of
    training iterations, and a single evaluation pass over the validation set. It implements the :meth:`~flambe.learn.Trainer.metric`
    method, which points to the best metric observed so far.

:class:`~flambe.learn.Evaluator`
    An :class:`~flambe.learn.Evaluator` evaluates a given :class:`~flambe.nn`Module` over a :class:`~flambe.dataset.Dataset` and computes given metrics.

:class:`~flambe.learn.Script`
    A :class:`~flambe.learn.Script` integrate a pre-written script with Flambé.

.. important::
    For more detailed information about this ``Components``, please refer to their documentation.

.. _understanding-component-subclassing_label:

Custom Component
---------------------

Custom ``Components`` should implement the :meth:`~flambe.compile.Component.run` method.
This method performs a single computation step, and returns a boolean,
indicating whether the ``Component`` is done executing (``True`` iff there is more work to do).

.. code-block:: python
    :linenos:

    class MyClass(Component):

        def __init__(self, a, b):
            super().__init__()
            ...

        def run(self) -> bool:
            ...
            return continue_flag

.. tip::
  We recommend always extending from an implementation of ``Component`` rather
  than implementing the plain interface. For example, if implementing an autoencoder,
  inherit from ``Module`` or if implementing cross validation training, inherit from ``Trainer``.

If you would like to include custom state in the state returned by :meth:`~flambe.compile.Component.get_state` method
see the :ref:`understanding-experiments-custom-state_label` section and the :class:`~flambe.compile.Component` package reference.

Then in YAML you could do the following:

.. code-block:: yaml

    !MyClass
      a: val1
      b: val2

    # or using the registrable_factory

Flambé also provides a way of registering factory methods to be used in YAML:

.. code-block:: python
    :linenos:

    class MyClass(Component):

        ...

        @registrable_factory
        @classmethod
        def special_factory(cls, x, y):
            a, b = do_something(x, y)
            return cls(a, b)


Now you can do:

.. code-block:: yaml

    !MyClass.special_factory
      x: val1
      y: val2

For information on how to add your custom :class:`~flambe.compile.Component` in the YAML files, go to :ref:`understanding-extensions_label`
