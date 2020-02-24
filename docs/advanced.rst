.. _understanding-advanced_label:

========
Advanced
========

Developer Mode
--------------

By using ``pip install -e .``, you enable the developer mode, which allows you to use Flambé
in `editable mode <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_.

By installing in developer mode (see :ref:`starting-install-dev_label`) Flambé will
automatically use the current code in your local copy of the repo that you installed,
including remote tasks.


Cache Git-based Extensions
--------------------------

As explained in :ref:`understanding-automatic-install_label`, flambé will install the extensions when ``-i``
is specified.

For all extensions that are git based URLs (from GitHub or BitBucket for example), flambé will clone the repositories into
``~/.flambe/extensions/`` folder the first time those extensions are being used. After this, every time one of those extensions
is being used flambé will pull instead of cloning again.

This allows ``Runnables`` to install extensions much faster in case they are heavy sized git repos.

.. attention::
  Flambé will warn once the size of ``~/.flambe/extensions/`` get bigger than 100MB.

.. _advanced-debugging_label:


Custom YAML Tags
-----------------

.. _understanding-advanced-yaml-alias_label:

Aliases
*******

Sometimes the best name for a class isn't the best or most convenient name to
use in a YAML config file. We provide an :func:`~flambe.compile.alias` class decorator
that can give your class alternative aliases for use in the config.

**Usage**

.. code-block:: python

    from flambe.compile import alias

    @alias('cool_tag')
    class MyClass(...):
        ...

Then start using your class as ``!cool_tag`` instead of ``MyClass`` in the config. Both options will still work though.
This combines seemlessly with extensions namespaces; if your extension's module name is "ext" then the new alias will
be ``!ext.cool_tag``.

.. _understanding-advanced-yaml-registrables_label:

Registrables
************

While you will normally subclass :class:`~flambe.compile.Component` to use some
class in a YAML configuration file, there may be situations where you don't want
all the functionality described in :ref:'understanding-component_label' such as
delayed initialization, and recursive compilation. For these situations you can
instead subclass the :class:`~flambe.compile.Registrable` class which only defines
the necessary functionality for loading and dumping into YAML. You will have
to implement your own :meth:`~flambe.compile.Registrable.from_yaml` and
:meth:`~flambe.compile.Registrable.to_yaml` methods.


**Example**

Let's say you want to create a new wrapper class around an integer that tracks
its name and a history of its values. First you would have to write your class

.. code-block:: python

    from flambe.compile import Registrable

    class SmartInt(Registrable):

        def __init__(self, name: str, initial_value: int):
            self.name = name
            self.initial_value = initial_value  # For dumping later
            self.val = initial_value

        ...  # Rest of implementation here

Then you'll want to implement your ``from_yaml`` and ``to_yaml`` in a way that makes sense
to you. Here, let's say the name and initial value should be separated by a dash
character:

.. code-block:: python

        @classmethod
        def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
            str_rep = f"{self.name}-{self.val}"
            representer.represent_str(tag, str_rep)

        @classmethod
        def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> Any:
            str_rep = constructor.construct_str(node)
            name, initial_value = str_rep.split()
            return cls(name, initial_value)

Finally you can now use your new Registrable object in YAML.

.. code-block:: yaml

    !Pipeline
    ...
    pipeline:
      stage_0: !Trainer
        param: !SmartInt my_param-9

.. attention:: You will need to make sure your code is part of an extension so that Flambé knows about your new class. See :ref:`understanding-extensions_label`

.. seealso:: The official `ruamel.yaml documentation <https://yaml.readthedocs.io/en/latest/>`_ for information about ``from_yaml`` and ``to_yaml``

.. seealso:: :class:`~flambe.compile.MappedRegistrable` can be referenced as another example or used if you just want a basic ``Registrable`` that can load from a dictionary of kwargs but doesn't have the other features of :class:`~flambe.compile.Component` like delayed initialization
