.. _understanding-extensions_label:

==========
Extensions
==========

Flambé comes with many built-in :class:`~flambe.compile.Component` and :class:`~flambe.runnable.Runnable` implementations.
(for example :class:`~flambe.experiment.Experiment`, :class:`~flambe.learn.Trainer`,
:class:`~flambe.dataset.TabularDataset` and :class:`~flambe.sampler.BaseSampler`). However, users
will almost always need to write new code and include that in their executions.
This is done via our **"extensions"** mechanism.

.. tip::
  At this point you should be familiar with the concepts of :ref:`understanding-component_label`
  and :ref:`understanding-runnables_label`.
    

.. important::
    The same "extensions" mechanism serves for importing custom ``Components`` and ``Runnables``.


.. _understanding-extensions-building_label:

Building Extensions
--------------------

**An extension is a pip installable package that contains valid Flambé objects as
top level imports.** These objects could be ``Runnables`` or ``Components``.
See `here <https://packaging.python.org/tutorials/packaging-projects/>`_
for more instructions on Python packages ready to be pip-installed.

Let's assume we want to define a custom trainer called ``MyCustomTrainer`` that
has a special behavior not implemented by the base ``Trainer``. The **extension** could have the following structure:

::

    extension
    ├── setup.py
    └── my_ext
        ├── my_trainer.py  # Here lives the definition of MyCustomTrainer
        └── __init__.py


.. code-block:: python
    :caption: extension/setup.py
    :linenos:

    from setuptools import setup, find_packages

    setup(
        name='my_extension-pkg-name',
        version='1.0.0',
        packages=find_packages(),  # This will install my_ext package
        install_requires=['extra_dependency==1.2.3'],
    )

.. code-block:: python
    :caption: extension/my_ext/__init__.py
    :linenos:

    from my_ext.my_trainer import MyCustomTrainer

    __all__ = ['MyCustomTrainer']


.. code-block:: python
    :caption: extension/my_ext/my_trainer.py
    :linenos:

    from flambe.learn import Trainer

    class MyCustomTrainer(Trainer):

        ...

        def run(self):
              # Do something special here

.. attention::
  If the extension was correctly built you should be able to ``pip install`` it and execute
  ``from my_ext import import MyCustomTrainer``, which means that this object is at the top level import.

.. _understanding-extensions-usage_label:

Using Extensions
-----------------

You are able to use any extension in any YAML config by specifying it in the
``extensions`` section which precedes the rest of the YAML:

.. code-block:: YAML

    my_extension: /path/to/extension
    ---
    !Experiment
    ... # use my_extension.MyCustomTrainer and other objects here

Each extension is declared using a ``key: value`` format.

.. important::
    **The** ``key`` **should be the top-level module name (not the package name)**.

The ``value`` can be:

* a local path pointing to the extension's folder (like in the above example)
* a remote GitHub repo folder URLs.
* a PyPI package (alongside its version)

For example:

.. code-block:: YAML

    my_extension: /path/to/extension
    my_other_extension: https://github.com/user/my_other_extension
    another_extension: py-extensions==1.0
    ---
    !Experiment

    ... # use my_extension.MyCustomTrainer and other objects here

Once an extension was added to the ``extensions`` section, all the extension's
objects become available using the module name as a prefix:


.. code-block:: YAML

    my_extension: /path/to/extension
    my_other_extension: https://github.com/user/my_other_extension
    ---
    pipeline:
        ...

        some_stage: !my_extension.MyCustomTrainer
           ...

        other_stage: !my_other_extension.AnotherCustomObject
           ...

.. important::
    Remember to use the **module name** as a prefix

.. hint:: **We support branches in GitHub extension repositories!** Just use ``https://github.com/user/repo/tree/<BRANCH_NAME>/path/to/extension``.

.. tip::
  Using extensions is similar to Python ``import`` statements. At the top of the file, you declare the
  non-builtin structures that you wish to use later.

    +---------------------------------------------+---------------------------------------------+
    | Python                                      | Flambe YAML                                 |
    +=============================================+=============================================+
    | .. code-block:: python                      | .. code-block:: yaml                        |
    |                                             |                                             |
    |   from my_extension  import MyCustomTrainer |   my_extension: /path/to/extensions         |
    |                                             |   ---                                       |
    |   ...                                       |   ...                                       |
    |   MyCustomTrainer(...)                      |   !my_extension.MyCustomTrainer             |
    |                                             |     ...                                     |
    +---------------------------------------------+---------------------------------------------+

