=================
Welcome to Flambé
=================


Flambé is a Python framework built to accelerate the development of machine learning research.

With Flambé you can:

* **Run hyperparameter searches** over arbitrary Python objects or scripts.
* **Constuct DAGs**, by searching over hyperparameters and reducing to the
best variants at any of the nodes.
* Distribute tasks **remotely** and **in parallel** over a cluster, with full AWS,
GCP, and Kubernetefs integrations.
* Easily **share** experiment configurations, results, and checkpoints with others.e
* Automate the **boilerplate code** in training models with `PyTorch <https://pytorch.org>`

Visit our website: https://flambe.ai
Visit out github repo: https://github.com/asappresearch/flambe  


**Getting Started**

Check out our :ref:`Installation Guide <starting-install_label>` and :ref:`_runnables` sections to get up and
running with Flambé in just a few minutes!

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   starting/motivation
   starting/install
   starting/contribute

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Walkthrough
  
   understanding/runnables
   understanding/searchable
   understanding/remote
   understanding/advanced

.. toctree::
   :titlesonly:
   :hidden:
   :caption: Tutorials

   tutorials/script
   tutorials/trainer
   tutorials/search
   tutorials/experiment
   tutorials/export
   tutorials/aws
   tutorials/kubernetes

.. toctree::
   :titlesonly:
   :hidden:
   :caption: Package Reference

   autoapi/flambe/dataset/index
   autoapi/flambe/cluster/index
   autoapi/flambe/compile/index
   autoapi/flambe/dataset/index
   autoapi/flambe/experiment/index
   autoapi/flambe/export/index
   autoapi/flambe/field/index
   autoapi/flambe/learn/index
   autoapi/flambe/logging/index
   autoapi/flambe/metric/index
   autoapi/flambe/model/index
   autoapi/flambe/nlp/index
   autoapi/flambe/nn/index
   autoapi/flambe/runner/index
   autoapi/flambe/sampler/index
   autoapi/flambe/tokenizer/index
   autoapi/flambe/vision/index
