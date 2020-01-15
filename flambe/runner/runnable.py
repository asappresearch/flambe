from abc import abstractmethod
from typing import List, Any, Optional

from flambe.compile import Registrable


class Environment(Registrable):
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
    remote: bool, optional
        Whether the envrionment is remote.
    orchestrator_ip: str, optional
        The orchestrator visible IP for the factories (usually
        the private IP)
    factories_ips: List[str], optional
        The list of factories IPs visible for other factories and
        orchestrator (usually private IPs)

    """

    def __init__(self,
                 ouput_path: str = 'flambe__output',
                 remote: bool = False,
                 orchestrator_ip: Optional[str] = None,
                 factories_ips: Optional[List[str]] = None) -> None:
        """Initialize the environment."""
        self.ouput_path = ouput_path
        self.remote = remote
        self.orchestrator_ip = orchestrator_ip
        self.factories_ips = factories_ips

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Use representer to create yaml representation of node"""
        kwargs = {'output_path': node.output_path,
                  'remote': node.remote,
                  'orchestrator_ip': node.orchestrator_ip,
                  'factories_ips': node.factories_ips}
        return representer.represent_mapping(tag, kwargs)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> Any:
        """Use constructor to create an instance of cls"""
        # NOTE: construct_yaml_map is a generator that yields the
        # constructed data and then updates it
        kwargs, = list(constructor.construct_yaml_map(node))
        return cls(**kwargs)


class Runnable(Registrable):
    """Abstract runnable interface."""

    @abstractmethod
    def run(env: Optional[Environment] = None):
        """Execute the Runnable.

        Parameters
        ----------
        env : Optional[Environment], optional
            An optional environment object.

        """
        pass
