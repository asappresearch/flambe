.. _Remote:

======
Remote
======

Flambé supports running remote ``Runnables`` where jobs can be distributed across a cluster of workers.
Flambé's cluster management is built on top of the `Ray Autoscaler <https://ray.readthedocs.io/en/latest/autoscaling.html>`_
and provides useful utilities to manage active clusters and their respective jobs.
For more information on particular cloud provider, see the sections below:

.. toctree::
   :titlesonly:
  
   clusters/aws
   clusters/gcp
   clusters/kubernetes

The full list of flambe commands is provided below:

+-------------------+--------------------------------------------------------------+
| Runnable          | Description                                                  |
+===================+==============================================================+
| :ref:`Run`        | Execute a runnable config.                                   |
+-------------------+--------------------------------------------------------------+
| :ref:`Submit`     | Submit a job to the cluster, as a YAML config.               | 
+-------------------+--------------------------------------------------------------+
| :ref:`Up`         | Launch / update a cluster.                                   |
+-------------------+--------------------------------------------------------------+ 
| :ref:`Down`       | Take down the cluster, optionally destroy it permanently     |
+-------------------+--------------------------------------------------------------+
| :ref:`Attach`     | Attach to a running job (i.e tmux session) on the cluster.   | 
+-------------------+--------------------------------------------------------------+
| :ref:`Ls`         | List the jobs (i.e tmux sessions) running on the cluster.    |
+-------------------+--------------------------------------------------------------+ 
| :ref:`Kill`       | Kill a job (i.e tmux session) running on the cluster.        |
+-------------------+--------------------------------------------------------------+
| :ref:`Clean`      | Clean the artifacts of a job on the cluster.                 | 
+-------------------+--------------------------------------------------------------+
| :ref:`Exec`       | Execute a command on the cluster head node.                  |
+-------------------+--------------------------------------------------------------+ 
| :ref:`Rsync-down` | Download files from the cluster.                             |
+-------------------+--------------------------------------------------------------+
| :ref:`Rsync-up`   | Upload files to the cluster.                                 | 
+-------------------+--------------------------------------------------------------+
| :ref:`Site`       | Launch a Web UI to monitor the activity on the cluster.      |
+-------------------+--------------------------------------------------------------+ 

For more information, see:

.. toctree::
   :titlesonly:
  
   commands
