from abc import abstractmethod
from flambe.runner import Runnable


class Searchable(Runnable):
    """Base Task interface.

    Tasks are at the core of Flambé. They are the inputs to both
    the ``Search`` and ``Experiment`` objects. A task can implemented
    with two simple methods:

    - ``step``: executes computation in steps. Returns a boolean
        indicating whether execution should continue or end.
    - ``metric``: returns a float used to compare different tasks's
        performance. A higher number should mean better.

    """

    @abstractmethod
    def step(self) -> bool:
        """Run a single computational step.

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
        pass

    @abstractmethod
    def metric(self) -> float:
        """Override this method to enable scheduling and searching.

        This method is called after every call to ``step``, and should
        return a unique scalar representing the current performance on
        the task, and to compare against other variants of the task.

        Returns
        -------
        float
            The metric to compare different variants of your task.

        """
        pass

    def run(self) -> float:
        """Execute the runnable.

        Returns
        -------
        float
            The metric to compare different variants of your task.

        """
        _continue = True
        while _continue:
            _continue = self.step()
        return self.metric()
