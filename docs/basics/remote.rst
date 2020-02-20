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
  
   remotes/aws
   remotes/gcp
   remotes/kubernetes

The full list of commands is given below:

  * attach      Attach to a running job (i.e tmux session) on the cluster.
  * clean       Clean the artifacts of a job on the cluster.
  * down        Take down the cluster, optionally destroy it permanently.
  * exec        Execute a command on the cluster head node.
  * kill        Kill a job (i.e tmux session) running on the cluster.
  * ls          List the jobs (i.e tmux sessions) running on the cluster.
  * rsync-down  Download files from the cluster.
  * rsync-up    Upload files to the cluster.
  * run         Execute a runnable config.
  * site        Launch a Web UI to monitor the activity on the cluster.
  * submit      Submit a job to the cluster, as a YAML config.
  * up          Launch / update the cluster.
