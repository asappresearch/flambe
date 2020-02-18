.. _tutorials-script_label:

==============================
Converting a script to Flambé
==============================

In many situations, you may already have a script that performs your full training routine,
and you would like to avoid doing any intergation work to leverage some of the tools in Flambé,
namely launching variants of the script on a large cluster 

For this particular use case, Flambé offers a ``Script`` component, which takes as input
the path to your scripts, and the relevant arguments. The script should read arguments
from ``sys.argv``, which means that traditional scripts, and scripts that use tools such
as ``argparse`` are natively supported.


Writing a config file
---------------------

You can define a configuration for your script using the ``Script`` runnable:

.. code-block:: yaml

    !Envrionment

    resources:
        script: '/path/to/your/script.py'

    ---

    !Script

    script: !r script 
    args: ['arg1', 'arg2']
    kwargs:
        kwarg1: !g [1, 5]  # Run a grid search over any arguments to your script
        kwarg2: 'foo'

That's it! You can now execute this configuration file, with the regular command:

.. code-block:: bash
    flambe run config.yaml

And execute it on a cluster with:

.. code-block:: bash
    flambe submit config.yaml [-c cluster.yaml]

In order to see tensorboard logs, simply import the logger, and use it anywhere in your script:

.. code-block:: bash
    from flambe import log
