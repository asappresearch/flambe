from typing import Optional

from flambe.compile.registered_types import RegisteredStatelessMap
from flambe.runner.environment import Environment


class Runnable(RegisteredStatelessMap):
    """Abstract runnable interface.

    Searchable are at the core of Flambé. They are the inputs to both
    the ``Search`` and ``Experiment`` objects. A task can implemented
    with two simple methods:

    - ``step``: executes computation in steps. Returns a boolean
        indicating whether execution should continue or end.
    - ``metric``: returns a float used to compare different tasks's
        performance. A higher number should mean better.

    """

    def run(self, environment: Optional[Environment] = None):
        """Implement this method to execute this object.

        Parameters
        ----------
        environment : Environment, optional
            An optional environment object.

        """
        raise NotImplementedError

    def step(self) -> bool:
        """Implement this method to enable hyperparameter search.

        When used in an experiment, this computational step should
        be on the order of tens of seconds to about 10 minutes of work
        on your intended hardware; checkpoints will be performed in
        between calls to run, and resources or search algorithms will
        be updated. If you want to run everything all at once, make
        sure a single call to run does all the work and return False.

        Returns
        -------
        bool
            True if should continue running later i.e. more work to do

        """
        raise NotImplementedError

    def metric(self) -> float:
        """Implement this method to enable hyperparameter search.

        This method is called after every call to ``step``, and should
        return a unique scalar representing the current performance,
        which is used to compare against other variants.

        Returns
        -------
        float
            The metric to compare different variants of your searchable.

        """
        raise NotImplementedError
