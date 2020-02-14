import copy
from typing import Any, Optional, Dict, List

from flambe.compile import Registrable, YAMLLoadType


class Environment(Registrable):
    """Hold information to use during execution.

    An Environment is simply a mapping, containing information
    such has ip addresses, extensions, and resources.  It is
    instantiated by the main Flambe process, and passed down to
    the run method of the objects following the Runnable interface.

    """

    def __init__(self,
                 output_path: str = 'flambe_output',
                 extensions: Optional[Dict[str, str]] = None,
                 local_resources: Optional[Dict[str, str]] = None,
                 remote_resources: Optional[Dict[str, str]] = None,
                 head_node_ip: Optional[str] = None,
                 worker_node_ips: Optional[List[str]] = None,
                 remote: bool = False,
                 debug: bool = False,
                 extra: Optional[Dict[str, Any]] = None) -> None:
        """Initialize an Environment.

        Parameters
        ----------
        output_path: str, optional
            The directory where to store outputs.
            Default ``flambe__output``
        extensions : Optional[Dict[str, str]], optional
            [description], by default None
        local_resources : Optional[Dict[str, str]], optional
            [description], by default None
        remote_resources : Optional[Dict[str, str]], optional
            [description], by default None
        head_node_ip: str, optional
            The orchestrator visible IP for the factories (usually
            the private IP)
        worker_node_ips : Optional[List[str]], optional
            [description], by default None
        debug : bool, optional
            [description], by default False
        extra : Optional[Dict[str, Any]], optional
            [description], by default None

        """
        self.output_path = output_path
        self.extensions = extensions or dict()
        self.local_resources = local_resources or dict()
        self.remote_resources = remote_resources or dict()
        self.head_node_ip = head_node_ip
        self.worker_node_ips = worker_node_ips or []
        self.remote = remote
        self.debug = debug
        self.extra = extra

        # TODO: remove this hack
        self._saved_arguments = {
            'output_path': self.output_path,
            'extensions': self.extensions,
            'local_resources': self.local_resources,
            'remote_resources': self.remote_resources,
            'head_node_ip': self.head_node_ip,
            'worker_node_ips': self.worker_node_ips,
            'remote': self.remote,
            'debug': self.debug,
            'extra': self.extra,
        }

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
        arguments = copy.deepcopy(self._saved_arguments)
        arguments.update(kwargs)
        return Environment(**arguments)  # type: ignore

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.KWARGS
