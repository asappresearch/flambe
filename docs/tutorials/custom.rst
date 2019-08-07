===========================
Using Custom Code in Flambé 
===========================

While Flambé offers a large number of `Component` objects to use in experiments,
researchers will typically need to use their own code, or modify one of our current
component object.


Writing your custom code
------------------------

Flambé configurations support any python object that inherits from ``flambe.Component``.
You can decide to inherit from one of our base classes such as ``flambe.nn.Module`` or ``flambe.dataset.Dataset``,
or you can inherit from ``flambe.Component`` directly.

A ``Component`` must simply implement a ``run`` method which returns a boolean indicating whether execution
should continue or not (useful for multi-step components such as a ``Trainer``).

Additionally, if you would like to run hyperparameter search on your custom
component, you must implement the ``metric`` method which returns the current best metric.


Setting up your extension
-------------------------

Flambé offers a simple mechanism to inject custom code in configs. To do so,
your code must be wrapped in a pip installable. If you are not familiar with
python packages, you can follow these 3 simple steps:

1. Make sure all your code is organised in a python module. In the example below,
   ``my_module`` is the name of the module containing our custom model and dataset objects.

::

    my_module # The name of your module
    ├── model.py
    ├── dataset.py
    ...

2. Next, we wrap the module in another folder, representing our package.

::

    my_package # This is the name of your package
    └── my_module # This is the name of your module
        ├── __init__.py
        ├── model.py
        ├── dataset.py
        └── ...

3. Finally, we write our setup.py file, which will make our package installable.
   This file is crucial as it indicates the external dependencies of your custom code.
   Below is a template for your setup file.

.. code-block:: python

    from setuptools import setup, find_packages

    setup(
        name = 'my_package', # the name of your package
        version = '1.0.0', # the version of your extension (optional)
        packages = find_packages(),
        install_requires = ['numpy >= 1.11.1', 'matplotlib >= 1.5.1'], # Idicate dependencies here
    )

After you add the setup file to your package, your final folder structure should look like
the one below:

::

    my_package
    ├── setup.py # This file makes your package installable
    └── my_module
        ├── __init__.py
        ├── model.py
        ├── dataset.py
        └── ...


Using your extension
--------------------

You have built your first extension! You can now use it freely in any configuration,
whether that'd be for an ``Experiment``, a ``Cluster`` or any other ``Runnable``.

To do so, simply add them at the top of your extension, mapping the name of the
module your built (``my_module`` in the example) to the location of the package
(``my_package`` in the example).

.. important:: The name of the module is used as a prefix in your configurations. Not
               the name of your package.

.. code-block:: yaml

    my_module: path/to/my_package  # Must map from module to package path

    ---  # Note the 3 dashes here

    !Experiment

    pipeline:
        dataset: !my_module.MyDataset # We use the name of your custom module as prefix

.. tip:: The path to the package may be a local path, a github URL, or the name of package one
          pypi. The latter allows you to specify a specific version of your extension. For github,
          we also support links to specific commit or branches.

Flambé will require your extension to be installed. You can do so manually by running:

``pip install my_package`` 

or Flambé can install all the extensions specified in your
configuration automatically if the ``-i`` flag is used:

``flambe -i config.yaml``
