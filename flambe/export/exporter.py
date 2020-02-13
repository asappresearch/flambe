from typing import Dict, Any, Optional

from flambe import Component
from flambe.runner import Environment


class Exporter(Component):
    """Implement an Exporter computable.

    This object can be viewed as a dummy computable. It is useful
    to group objects into a block when those get save, to more
    easily refer to them later on, for instance in an object builder.

    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """Initialize the Exporter.

        Parameters
        ----------
        kwargs: Dict[str, Any]
            Mapping from name to any object to export

        """
        self.objects = kwargs

        for name, obj in kwargs.items():
            setattr(self, name, obj)
            if not isinstance(obj, Component):
                self.register_attrs(name)

    def metric(self, env: Optional[Environment] = None) -> float:
        """Override this method to enable scheduling.

        Returns
        -------
        float
            The metric to compare computable varients

        """
        return 0

    def step(self, env: Optional[Environment] = None) -> bool:
        """Run the exporter.

        Returns
        -------
        bool
            False, as this is a single step Component.

        """
        return False
