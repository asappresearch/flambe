from typing import Dict

from flambe import Component


class Exporter(Component):
    """Implement an Exporter computable.

    This object can be viewed as a dummy computable. It is useful
    to group objects into a block when those get save, to more
    easily refer to them later on, for instance in an object builder.

    """

    def __init__(self, **kwargs: Dict[str, Component]) -> None:
        """Initialize the Exporter.

        Parameters
        ----------
        kwargs: Dict[str, Component]
            Mapping from name to Component object

        """
        self.objects = kwargs

        for name, obj in kwargs.items():
            setattr(self, name, obj)

    def run(self) -> bool:
        """Run the exporter.

        Returns
        -------
        bool
            False, as this is a single step Component.

        """
        return False
