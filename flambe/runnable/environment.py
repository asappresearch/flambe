from typing import List, Any
from flambe.compile import Registrable


class RemoteEnvironment(Registrable):
    """This objects contains information about the cluster

    This object will be available on the remote execution of
    the ClusterRunnable (as an attribute).

    IMPORTANT: this object needs to be serializable, hence it Needs
    to be created using 'compile' method.

    Attributes
    ----------
    key: str
        The key that communicates the cluster
    orchestrator_ip: str
        The orchestrator visible IP for the factories (usually
        the private IP)
    factories_ips: List[str]
        The list of factories IPs visible for other factories and
        orchestrator (usually private IPs)
    user: str
        The username of all machines. This implementations assumes
        same usename for all machines

    """

    def __init__(self,
                 key: str,
                 orchestrator_ip: str,
                 factories_ips: List[str],
                 user: str,
                 **kwargs) -> None:
        self.key = key
        self.orchestrator_ip = orchestrator_ip
        self.factories_ips = factories_ips
        self.user = user

    @classmethod
    def to_yaml(cls, representer: Any, node: Any, tag: str) -> Any:
        """Use representer to create yaml representation of node"""
        kwargs = {'key': node.key,
                  'orchestrator_ip': node.orchestrator_ip,
                  'factories_ips': node.factories_ips,
                  'user': node.user}
        return representer.represent_mapping(tag, kwargs)

    @classmethod
    def from_yaml(cls, constructor: Any, node: Any, factory_name: str) -> Any:
        """Use constructor to create an instance of cls"""
        # NOTE: construct_yaml_map is a generator that yields the
        # constructed data and then updates it
        kwargs, = list(constructor.construct_yaml_map(node))
        return cls(**kwargs)
