from typing import Any, Optional, Dict

from flambe.compile.registered_types import RegisteredStatelessMap


class Environment(RegisteredStatelessMap):
    """This objects contains information about the cluster

    This object will be available on the remote execution of
    the ClusterRunnable (as an attribute).

    IMPORTANT: this object needs to be serializable, hence it Needs
    to be created using 'compile' method.

    Attributes
    ----------
    output_path: str, optional
        The directory where to store outputs.
        Default ``flambe__output``
    head_node_ip: str, optional
        The orchestrator visible IP for the factories (usually
        the private IP)

    """

    def __init__(self,
                 output_path: str = 'flambe_output',
                 extensions: Optional[Dict[str, str]] = None,
                 resources: Optional[Dict[str, str]] = None,
                 head_node_ip: Optional[str] = None,
                 debug: bool = False) -> None:
        """Initialize the environment."""
        self.output_path = output_path
        self.extensions = extensions
        self.resources = resources
        self.head_node_ip = head_node_ip
        self.debug = debug

    def clone(self, **kwargs) -> 'Environment':
        """Clone the envrionment, updated with the provided arguments.

        Parameters
        ----------
        **kwargs: Dict
            Arguments to override.

        Returns
        -------
        Environment
            The new updated envrionement object.

        """
        arguments = {
            'output_path': self.output_path,
            'extensions': self.extensions,
            'resources': self.resources,
            'head_node_ip': self.head_node_ip,
            'debug': self.debug
        }
        arguments.update(kwargs)
        return Environment(**arguments)  # type: ignore

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Use representer to create yaml representation of node"""
        kwargs = {'output_path': node.output_path,
                  'extensions': node.extensions,
                  'resources': node.resources,
                  'head_node_ip': node.head_node_ip,
                  'debug': node.debug}
        return representer.represent_mapping(tag, kwargs)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str, tag: str) -> Any:
        """Use constructor to create an instance of cls"""
        # NOTE: construct_yaml_map is a generator that yields the
        # constructed data and then updates it
        kwargs, = list(constructor.construct_yaml_map(node))
        return cls(**kwargs)
