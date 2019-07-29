===========================
Using Custom Code in Flambé 
===========================

While Flambé offers a large number of `Component` objects to use in experiments,
researchers will tipically need to use their own code, or modify one of our current
component object.

Flambé offers a simply mechanism to inject custom code in configs. Specifically,
your code should be wrapped in a pip installable. This is as simple as including
a setup.py alongside your module. For example:

Say you have the following directory structure for your project:

::

    my_project
    ├── model.py
    ├── dataset.py
    ...

The first step is to convert your project into a pip installable.

::

    my_pip_installable # This is the name of your package, doesn't matter
    ├── setup.py
    └── my_project # This is the name of your module, this is what you will use in the config
        ├── __init__.py
        ├── model.py
        ├── dataset.py
        └── ...

The reason behind using pip installables is for you to indicate external library requirements
in your setup.py under ``install_requires``.

Once that is done, all you need to do is make sure that you inherit from one of our base classes
such as ``flambe.nn.Module`` or ``flambe.nn.Dataset``. Alternativly you can also inherit from
the ``flambe.Component`` object directly.

A Component must implement a ``run`` method which returns a boolean indicating whether execution
should continue or not (useful for multi-step components such as a ``Trainer``).

.. attention:: Make sure that all your components are surfaced at the top level __init__, as This
is how Flambé will register them for config usage.

You have now built your first extension! You can now use it freely in any configuration,
whether that'd be for an ``Experiment``, a ``Cluster`` or any other ``Runnable``:


.. code-block:: yaml

    my_project: /path/to/my_pip_installable/  # Must map from module to package location
    ---

    Experiment

    pipeline:
        dataset: !my_project.MyDataset # We use the name of your custom module as prefix

.. tip:: The path to the package may be a local path, a github URL, or the name of package one
          pipy. The latter allows you to specify a specific version of your extenion. For github,
          we also support links to specific commit or branches.
