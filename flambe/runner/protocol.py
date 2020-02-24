from typing_extensions import Protocol, runtime_checkable

from flambe.compile.yaml import load_config_from_file


@runtime_checkable
class Runnable(Protocol):
    """Abstract runnable interface.

    Implementing this interface enables execution through the flambe
    ``run`` command. It only requires implementing a run method.

    """

    def run(self) -> bool:
        """Run a computational step, returns True until done.

        When used in a search, this computational step should
        be on the order of tens of seconds to about 10 minutes of work
        on your intended hardware; checkpoints will be performed in
        between calls to run, and resources or search algorithms will
        be updated. If you want to run everything all at once, make
        sure a single call to run does all the work and return False.

        Returns
        -------
        bool
            True until execution is over.

        """
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
