import os
import copy
from contextlib import contextmanager
from typing import Any, Optional, Dict, List, Iterator

from flambe.compile import Registrable, YAMLLoadType
from flambe.compile.yaml import load_first_config, load_config_from_file
from flambe.compile.yaml import dump_config, dump_one_config, num_yaml_files


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
                 local_files: Optional[Dict[str, str]] = None,
                 remote_files: Optional[Dict[str, str]] = None,
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
        local_files : Optional[Dict[str, str]], optional
            [description], by default None
        remote_files : Optional[Dict[str, str]], optional
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
        self.local_files = local_files or dict()
        self.remote_files = remote_files or dict()
        self.head_node_ip = head_node_ip
        self.worker_node_ips = worker_node_ips or []
        self.remote = remote
        self.debug = debug
        self.extra = extra

        # TODO: remove this hack
        if not hasattr(self, '_saved_arguments'):
            from ruamel.yaml.comments import CommentedMap
            saved_arguments = CommentedMap({
                'output_path': self.output_path,
                'extensions': self.extensions,
                'local_resources': self.local_resources,
                'remote_resources': self.remote_resources,
                'head_node_ip': self.head_node_ip,
                'worker_node_ips': self.worker_node_ips,
                'remote': self.remote,
                'debug': self.debug,
                'extra': self.extra,
            })
            saved_arguments.yaml_set_tag('!Environment')
            self._saved_arguments = saved_arguments

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


def get_env(**kwargs: Dict[str, Any]) -> 'Environment':
    """Get the global flambe environment and apply modifications.

    Parameters
    ----------
    kwargs: Dict[str, Any]
        Arguments to modify the global environment with.

    Returns
    -------
    Environment
        A copy of the global environment.

    """
    if 'FLAMBE_ENVIRONMENT' not in os.environ:
        env = Environment()
    else:
        env = load_first_config(os.environ['FLAMBE_ENVIRONMENT'])
    return env.clone(**kwargs)


def set_env(env: Optional['Environment'] = None, **kwargs: Dict[str, Any]):
    """Set the global flambe environment and apply modifications.

    Parameters
    ----------
    env: Environment, optional
        An optional environment to set as global
    kwargs: Dict[str, Any]
        Arguments to modify the global environment with.

    """
    if 'FLAMBE_ENVIRONMENT' not in os.environ:
        env = Environment() if env is None else env
    else:
        env = load_first_config(os.environ['FLAMBE_ENVIRONMENT']) if env is None else env
    new_env = env.clone(**kwargs)
    os.environ['FLAMBE_ENVIRONMENT'] = dump_one_config(new_env)  # type: ignore


@contextmanager
def env(env: Optional['Environment'] = None, **kwargs: Dict[str, Any]) -> Iterator['Environment']:
    """Context manager to fetch and temporarley modify the environment.

    Parameters
    ----------
    env: Environment, optional
        An optional environment to set as global
    kwargs: Dict[str, Any]
        Arguments to modify the global environment with.

    Returns
    -------
    Environment
        A copy of the environment

    """
    current_env = get_env()
    try:
        set_env(env=env, **kwargs)
        yield get_env()
    finally:
        set_env(env=current_env)


def load_env_from_config(path: str) -> Optional[Environment]:
    """Load an envrionment from the given config, if present.

    Parameters
    ----------
    path : str
        A path to the cluster config.

    Returns
    -------
    Cluster
        The loaded cluster object

    """
    if num_yaml_files(path) > 1:
        config = next(load_config_from_file(path))
        return config
    return None


def set_env_in_config(env: Environment, input_path: str, stream: Optional[Any] = None):
    """Set the environment.

    Parameters
    ----------
    path : str
        A path to the cluster config.

    Returns
    -------
    Cluster
        The loaded cluster object

    """
    configs = list(load_config_from_file(input_path, convert=False))
    dump_config([env, configs[-1]], stream)
