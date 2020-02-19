.. _runnables:

=========
Runnables
=========

Flambé executes ``Runnables``, which are simply Python objects that implement the method ``run``:

    .. code-block:: python

        class Runnable(object):

            def run(self):
                ...


Flambé provides the following set of ``Runnables``, but you can easily create your own:

+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Runnable          | Description                                                                                                                                                                                               |
+===================+===========================================================================================================================================================================================================+
| :ref:`Script`     | An entry-point for users who wish to keep their code unchanged, but leverage Flambé's cluster management and distributed hyperparameter search tools.                                                     |
+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Trainer`    | Train / Evaluate a single model on a given task. Offers an interface to automate the boilerplate code usually found in PyTorch scripts, such as multi-gpu handling, fp16 training, and training loops.    | 
+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`Search`     | Run a hyperparameter search over python objects and scriptsy                                                                                                                                              |
+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+ 
| :ref:`Experiment` | Build a computational DAG, with the possibility of running a hyperparameter search at each node, and reduce to the best variants                                                                          |
+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

``Runnables`` can be used in regular python scripts, or executed through YAML configurations with the command:

```bash
flambe run [CONFIG]
```

To see all the arguments to the ``run`` command use:

```bash
flambe run --help
```

To submit to a cluster:

    .. code-block:: bash
        
        flambe submit [CONFIG] --cluster ~/.flambe/cluster.yaml

For more information on remote execution, see: :ref:`Remote`.


**Environment**

When executing Flambé runnables, you can access an environment object containing
useful information about the execution, including:

* The output path to use
* A list of the external python modules required for execution
* A list of resources (i.e files and folders) required for execution
* The IP's of the machines executing the runnable

The envrionment object is used to manage remote execution, as well as to ensure reproducibility.

To fetch the envrionment object anywhere in your code, use:

.. code-block:: python

    import flambe

    ...

    env = flambe.get_env()

Note that you can also override any attribute on the envrionment by passing arguments
to the ``get_env`` function. You can also make these changes permanent by modifying
the global envrionment:

.. code-block:: python

    import flambe

    ...

    flambe.set_env(env=env, ...)

When executing the command line ``run`` or ``submit`` command, you may pass a custom envrionment,
either through the ``--env`` argument to the command line or by adding an extra section in your
YAML configuration:

.. code-block:: YAML

    !Environment

    ...
    ---
    !Runnable
    ...


**Specifying an extension**

The ``Environment`` object recieves dictionary argument names ``extensions``.
Each extension is declared using a ``key: value`` format where the key is the 
**the top-level module name (not the package name)**, and the ``value`` can be:

* a local path pointing to a folder or file containing the code to load
* a remote GitHub repo folder URLs.
* a PyPI package (alongside its version)

For example:

.. code-block:: YAML

    !Environment

    extensions:
        foo: /path/to/extension
    ---
    !foo.ACustomRunnable

    ...


.. hint:: **We support branches in GitHub extension repositories!** Just use ``https://github.com/user/repo/tree/<BRANCH_NAME>/path/to/extension``.

.. tip::
  Using extensions is similar to Python ``import`` statements. At the top of the file, you declare the
  non-builtin structures that you wish to use later.

    +---------------------------------------------+---------------------------------------+
    | Python                                      | Flambe YAML                           |
    +=============================================+=======================================+
    | .. code-block:: python                      | .. code-block:: yaml                  |
    |                                             |                                       |
    |                                             |   !Environment                        |
    |                                             |   extensions:                         |
    |   from my_extension  import MyCustomTrainer |     my_extension: /path/to/extensions |
    |                                             |   ---                                 |
    |                                             |                                       |
    |   MyCustomTrainer(...)                      |   !my_extension.MyCustomTrainer       |
    |                                             |     ...                               |
    +---------------------------------------------+---------------------------------------+


.. toctree::
   :titlesonly:
  
   runnables/script
   runnables/trainer
   runnables/search
   runnables/experiment
   runnables/export
