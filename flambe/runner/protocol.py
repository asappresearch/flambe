from typing import Optional
from typing_extensions import Protocol, runtime_checkable

from flambe.runner.environment import Environment


@runtime_checkable
class Runnable(Protocol):
    """Abstract runnable interface.

    Implementing this interface enables execution through the flambe
    ``run`` command. It only requires implementing a run method.
    The run method recieves an environment object which is used to
    provide information that varies depending on whether execution
    is local or remote (for example an output path)

    """

    def run(self, environment: Optional[Environment] = None):
        """Implement this method to execute this object.

        Parameters
        ----------
        environment : Environment, optional
            An optional environment object.

        """
        raise NotImplementedError
