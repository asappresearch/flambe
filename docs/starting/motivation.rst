.. _starting-motivation:

==========
Motivation
==========

Flambé's primary objective is to **speed up all of the research lifecycle** including model prototyping,
hyperparameter optimization and execution on a cluster.


Why Flambé?
-----------

1. Running machine learning experiments takes a lot of continuous and tedious effort.
2. Standardizing data preprocessing and weights sharing across the community or within a team is difficult.

We've found that while there are several new libraries offering a selection of
reliable model implementations, there isn't a great library that couples these
modules with an experimentation framework. Since experimentation (especially
hyper-parameter search, deployment on remote machines, and data loading and
preprocessing) is one of the most important and time-consuming aspects of ML
research we decided to build Flambé.

An important component of Flambé is Ray, an open source distributed ML library.
Ray has some of the necessary infrastructure to build experiments at scale;
coupled with Flambé you could be tuning many variants of your already existing
models on a large cluster in minutes! Flambé's crucial contribution is to
facilitate rapid iteration and experimentation where tools like Ray and AllenNLP
alone require large development costs to integrate.

The most important contribution of Flambé is to improve the user experience
involved in doing research, including the various phases of experimentation
we outlined at the very beginning of this page. To do this well, we try to
adhere to the following values:

Core values
-----------

- **Practicality**: customize functionality in code, and iterate over settings and hyperparameters in config files.
- **Modularity & Composability**: rapidly repurpose existing code for hyper-parameter optimization and new use-cases
- **Reproducability**: reproducible experiments, by anyone, at any time.
