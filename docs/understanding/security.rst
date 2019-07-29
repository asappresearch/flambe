.. _understanding-security_label:

========
Security
========

When creating clusters, sending information to instances or even pulling extensions
from GitHub, flambé needs to deal with user's protected data.

.. important::
  Flambé will always rely on the default local configuration users have
  for all services flambé uses. This means that when using services like github or AWS, flambé
  will rely on the standard authentication mechanisms each service requires.
  
  **There is no need for users to configure special auth mechanisms for flambé**.

Secrets
-------

As explained in :ref:`understanding-providing-secrets_label`, users have the posibility of
providing secrets information to the :class:`~flambe.runnable.Runnable` objects that will be executed.

This is done via an `ini <https://en.wikipedia.org/wiki/INI_file>`_ file. For example:

.. code-block:: ini

    [SERVICE]
    SECRET_TOKEN = ABCDEFGHI123456789


    [OTHER]
    PASSWORD = 0987654321

When calling:

.. code:: bash

    flambe runnable.yaml --secrets secrets.ini [--cluster cluster.yaml]

Then the :class:`~flambe.runnable.Runnable` will have access to the secrets through its attribute
:attr:`~flambe.runnable.Runnable.config`.

.. _understanding-security-clusters_label:

Clusters
--------

Some important items related to Security when dealing with clusters:

* Flambé will use SSH for all communications with the instances. It will use **only**
  the key specified in the config.
* All resources that need to be uploaded to the instances are done via **rsync with the
  given key**.
* A copy of the secrets file will be sent to all instances using **rsync with the given key**.
* **The key provided in the config is never uploaded to the instances.**
* When loading the cluster, flambé will distribute a special key pair (created exclusively
  for the cluster) to all instances. This key will be used for internal communication (ie 
  the communication betweeen the instances)

.. important::
   All flow between the local process and the instances is done in a secure way using
   SSH protocols.


.. attention::
   Flambé won't configure any instances security policies (eg firewalls, security groups, etc).
   The user is responsible for configuring this to ensure clusters work correctly. This involves:

    * Allowing private communication between the hosts created in the same subnet.
    * Opening port 22 for SSH connection in all hosts.
    * Opening ports 49556 and 49558 for the Report Site (in case of running an :class:`~flambe.experiment.Experiment`).

AWS
***

When using :class:`~flambe.cluster.AWSCluster` for creating an AWS EC2 cluster, flambé will rely on the local configuration
for authentication (it uses ``boto3`` under the hood). Users are going to be able to create clusters as
long as they follow the AWS standards for storing credentials/tokens (for example, having ``~/.aws/credentials``
file or having ``AWS_*`` environment variables defined).

Extensions
----------

As explained in :ref:`understanding-automatic-install_label`, flambé will install the extensions when ``-i``
is specified.
For all extensions that are git based URLs (from GitHub or BitBucket for example), then **flambé will try to clone/pull
them using the local configuration**. This means that if for example a user wants to use an extensions from its private
GitHub account, then it needs to have local configuration that allows pulling from this GitHub account. Flambé will
not provide any special SSH keys to authenticate with these services.

.. hint::
  git URLs for extensions support both HTTPS/SSH protocols:

  .. code-block:: YAML

    extensions: ssh://git@github.com:user/repo.git
    other_extension: https://github.com/user/repo/tree/my_branch/extensions
    ---
    !Runnable
    ...

