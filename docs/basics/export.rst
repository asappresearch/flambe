========
Exporters
========

A :class:`~flambe.export.Exporter` is a simple runnable that can be used to construct a python
object by combining local or remote artifacts, and uploading the constructuted object
locally or remotely (on Amazon S3 for example).

``Exporters`` decouple the inference logic with the training logic, allowing users
to iterate through inference contracts without the need to rerun training.


Motivation
----------

Let's assume that a user wants to train a binary classifier using a :class:`~flambe.learn.Trainer`:

.. code-block:: yaml

    !Trainer
    ..
    model: !LogisticRegression


Now, the user needs to implement an inference object ``ClassifierEngine`` that has a
method ``predict`` that performs the forward pass on the trained ``model``:

.. code-block:: python
    :linenos:

    from flambe.nn import Module

    class ClassifierEngine(object):

       def __init__(self, model: Module):
          self.model = model

       def predict(self, **kwargs):
          # Custom code (for example, requests to APIs)
          p = self.model(feature)
          return {"POSITIVE": p, "NEGATIVE": 1-p}

The user can use a :class:`~flambe.export.Exporter` to build this object:

.. code-block:: yaml

    !Environment
    extensions:
        ext: /path/to/my/extensions
    ---
    !Exporter
    
    storage: s3
    destination: my-bucket

    ..
    obj: !ClassifierEngine
        ...
        model: !ext.MyBinaryClassifier.load_from_path:
          path: /path/to/saved/modeel

The inference object will be saved in ``s3://my-bucket``. Then the user can:

.. code-block:: python
    :linenos:

    import flambe

    inference_engine = flambe.load("s3://my-bucket")
    inference_engine.predict(...)
    # >> {"POSITIVE": 0.9, "NEGATIVE": 0.1}

.. important::
    Note that the inference logic is decoupled from the object that produced the Trainer. If in the
    future the inference logic changes, there is no need of rerunning it.


Example
-------


.. code-block:: yaml

    !Exporter
    
    storage: [ local | s3 ]
    destination: path/to/location

    ..
    component: !MyComponent
        params1: value1
        params2: value2
        ...
        paramsN: valueN


.. important::
    For a full list of parameters, go to :class:`~flambe.export.Exporter`.


.. hint::
    If storage is **"s3"**, then the destination can be an S3 bucket folder. Flambé will
    take care of uploading the built artifacts.
