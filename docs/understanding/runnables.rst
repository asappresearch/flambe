.. _understanding-runnables_label:

=========
Runnables
=========

Runnables are top level objects that flambé is able to run.
:class:`~flambe.experiment.Experiment` is an example of a :class:`~flambe.runnable.Runnable`.
 
Implementing Runnables is as easy as implementing the Runnable interface.
Esentially, it requires a single method :meth:`~flambe.runnable.Runnable.run`.
 
.. hint::
  A ``Runnable`` object can be executed by flambé: 

    .. code-block:: bash

         flambe runnable.yaml

For example, let’s imagine we want to implement an **S3Pusher** that takes
an ``Experiment`` output folder and uploads the content to a specific S3 bucket:
 
.. code-block:: python
    :linenos:

    from flambe.runnable import Runnable

    class S3Pusher(Runnable):

        def __init__(self, experiment_path: str, bucket_name: str) -> None:
            super().__init__()
            self.experiment_path = experiment_path
            self.bucket_name = bucket_name

        def run(self, **kwargs) -> None:
            """Upload a local folder to a S3 bucket"""

            # Code to upload to S3 bucket
            S3_client = boto3.client("s3")
            for root, dirs, files in os.walk(self.experiment_path):
                for f in files:
                    s3C.upload_file(os.path.join(root, f), self.bucketname, f)

This class definition can now be included in an extension (read more about extensions
in :ref:`understanding-extensions_label`) and used as a top level object in a YAML file.

.. code-block:: yaml

    ext: /path/to/S3Pusher/extension:
    ---

    !ext.S3Pusher
        experiment_path: /path/to/my/experiment 
        bucket_name: my-bucket

Then, simply execute:

.. code:: bash
 
     flambe s3pusher.yaml

.. _understanding-providing-secrets_label: 

Providing secrets
-----------------

All ``Runnables`` have access to secret information that the users can share via 
an `ini file <https://en.wikipedia.org/wiki/INI_file>`_.

By executing the ``Runnable`` with a ``--secrets`` parameter, then the ``Runnable`` can
access the secrets through the :attr:`~flambe.runnable.Runnable.config` attribute:

Let’s say that our ``S3Pusher`` needs to access the ``AWS_SECRET_TOKEN``. Then: 

.. code-block:: python
    :linenos:

    from flambe.runnable import Runnable

    class S3Pusher(Runnable):
        
        def __init__(self, experiment_path: str, bucket_name: str) -> None:
            # Same as before
            ...

        def run(self, **kwargs) -> None:
            """Upload a local folder to a S3 bucket"""

            # Code to upload to S3 bucket
            S3_client = boto3.client("s3", token=self.config['AWS']['AWS_SECRET_TOKEN'])
            for root, dirs, files in os.walk(self.experiment_path):
                for file in files:
                    s3C.upload_file(os.path.join(root, file), self.bucketname, file)

Then if ``secrets.ini`` contains:

.. code-block:: ini

    [AWS]
    AWS_SECRET_TOKEN = ABCDEFGHI123456789

We can execute:

.. code:: bash

  flambe s3pusher.yaml --secrets secret.ini

.. _understanding-extensions-install_label: 

Extensions installation
-----------------------

.. important::
    To understand this section you should be familiar with extensions. For information about
    extensions, go to :ref:`understanding-extensions_label`.

When executing a :class:`~flambe.runnable.Runnable`, it's possible that extensions are being involved. For example:

.. code-block:: yaml

    ext: /path/to/extension
    other_ext: http://github.com/user/some_extension
    ---

    !ext.CustomRunnable
        ...
        param: !other_ext.CustomComponent

These extensions need to be installed in order to run the ``Runnable``. Users can easily do this using ``pip``, as the
extensions are just python packages.

.. warning::
   The extensions section (as seen in the example above) contains a dictionary where the key is the module name
   and the value is the package. **The package is the one that the user needs to ``pip`` install.
   are being updated.**

Other flags
-----------

When executing flambé CLI passing a YAML file, this additional flags can be provided (among others):

``--verbose / -v``
    All logs will be displayed in the console.

``--force``
    ``Runnables`` may chose to accept this flag in its :meth:`~flambe.runnable.Runnable.run` method
    to provide some overriding policies.
