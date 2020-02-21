=================
Welcome to Flambé
=================


Flambé is a Python framework built to accelerate machine learning research. With Flambé you can:

* Run hyperparameter searches over arbitrary Python objects or scripts.
* Constuct DAGs to search over and reduce to the best hyperparameter variants at any node.
* Distribute tasks remotely over a cluster, with full AWS, GCP, and Kubernetes integrations.
* Easily share experiment configurations, results, and checkpoints with others.e
* Automate the boilerplate code in training models with `PyTorch <https://pytorch.org>`_

| Visit our website: https://flambe.ai  
| Visit out github repo: https://github.com/asappresearch/flambe  

.. toctree::
   :titlesonly: 
   :caption: Flambé

   starting/install
   starting/motivation
   starting/contribute

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
  
   basics/overview
   basics/script
   basics/trainer
   basics/search
   basics/experiment
   basics/export
   basics/cluster

.. toctree::
   :titlesonly:
   :caption:  Advanced

   advanced

.. toctree::
   :titlesonly:
   :caption: Package Reference

   autoapi/flambe/dataset/index
   autoapi/flambe/cluster/index
   autoapi/flambe/compile/index
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
