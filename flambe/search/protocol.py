from flambe.runner.protocol import Runnable


class Comparable(Runnable):  # type: ignore
    """The Comparable interface.

    Implementing this interface enables the use of an object with
    the flambe search module. Note that you do not need to inherit
    from this class. Instead, you must simply ensure that you have
    implements the following methods:

    - run: runs a computational step. We usually treat a step as a
      series of training iteration, and a validation step. The method
      should return a boolean indicating whether to continue executing.
    - metric: get the current validation metric. A higher number is
      considered better. If you happen to have a metric that improves
      as it gets smaller, you may return a negative number.

    Comparable's are at the core of Flambé. They are the expected
    inputs to both the ``Search`` and ``Pipeline`` objects.

    """

    def metric(self) -> float:
        """Return the current performance on the task.

        This method is called after every call to ``step``, and should
        return a unique scalar representing the current performance,
        which is used to compare against other variants.

        Returns
        -------
        float
            The metric to compare different variants of your searchable.

        """
        raise NotImplementedError
