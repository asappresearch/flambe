from typing_extensions import Protocol


class Searchable(Protocol):
    """The Searchable interface.

    Implementing this interface enables the use of an object with
    the flambe search module. Note that you do not need to inherit
    from this class. Instead, you must simply ensure that you have
    implements the following methods:

    - step: runs a computational step. We usually treat a step as a
      series of training iteration, and a validation step. The method
      should return a boolean indicating whether to continue executing.
    - metric: get the current validation metric. A higher number is
      considered better. If you happen to have a metric that improves
      as it gets smaller, you may return a negative number.

    Searchable's are at the core of Flambé. They are the expected
    inputs to both the ``Search`` and ``Experiment`` objects.

    """

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
