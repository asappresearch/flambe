from typing_extensions import Protocol, runtime_checkable

from flambe.compile.yaml import load_config_from_file


@runtime_checkable
class Runnable(Protocol):
    """Abstract runnable interface.

    Implementing this interface enables execution through the flambe
    ``run`` command. It only requires implementing a run method.

    """

    def run(self):
        """Implement this method to execute this object."""
        raise NotImplementedError


def load_runnable_from_config(path: str) -> Runnable:
    """Load a Cluster obejct from the given config.

    Parameters
    ----------
    path : str
        A path to the cluster config.
    convert: bool
        Convert to a python object at loading time.

    Returns
    -------
    Runnable
        The loaded cluster object if convert is True, otherwise
        its yaml representation.

    """
    configs = list(load_config_from_file(path))
    return configs[-1]
