from typing import Optional, Dict, List
from ruamel.yaml import YAML
import subprocess
import tempfile

from ray.autoscaler.commands import get_head_node_ip, get_worker_node_ips

from flambe.runner import Runnable
from flambe.compile import Registrable


class Cluster(Registrable):
    """Base cluster implementation."""

    def __init__(self,
                 name: str,
                 initial_workers: int = 1,
                 min_workers: int = 1,
                 max_workers: int = 1,
                 target_utilization_fraction: float = 0.8,
                 idle_timeout_minutes: int = 5,
                 ssh_user: str = 'ubuntu',
                 ssh_private_key: Optional[str] = None,
                 file_mounts: Optional[Dict[str, str]] = None,
                 setup_commands: Optional[List[str]] = None,
                 head_setup_commands: Optional[List[str]] = None,
                 worker_setup_commands: Optional[List[str]] = None) -> None:
        """Initialize a cluster.

        Parameters
        ----------
        name : str
            A name for the cluster.
        initial_workers: int
            The initial number of worker nodes to start.
        min_workers : int
            The minimum number of worker nodes to keep running.
        max_workers : int
            The maximum number of worker nodes to keep running.
        target_utilization_fraction: float
            The target utilization threshold before creating a node.
        idle_timeout_minutes: int
            The number of minutes to allow a node to be idle, before
            getting killed.
        ssh_user: str, optional
            Username to use to SSH into the nodes.
            IMPORTANT: all instances need to have the same username.
        ssh_private_key: str, optional
            The path to the ssh key used to communicate to all
            instances. IMPORTANT: all instances must be accessible
            with the same key.
        setup_commands: Optional[List[str]]
            A list of commands to run on all hosts for setup purposes.
            These commands can be used to mount volumes, install
            software, etc. Defaults to None.
            IMPORTANT: the commands need to be idempotent and
            they shouldn't expect user input
        head_setup_commands : Optional[List[str]], optional
            A list of commands to run on the head node only.
        worker_setup_commands : Optional[List[str]], optional
            A list of commands to run on the factory nodes only.

        """
        setup_commands = setup_commands or []
        setup_commands.append('pip install --user -U flambe')

        config = {
            'auth': {
                'ssh_user': ssh_user,
                'ssh_private_key': ssh_private_key
            },
            'file_mounts': file_mounts or {},
            'setup_commands': setup_commands,
            'head_setup_commands': head_setup_commands or [],
            'worker_setup_commands': worker_setup_commands or []
        }

        self.config = config

    def _run(self, command):
        """Update / Create the cluster."""
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            subprocess.run(f"ray {command} {fp.name}")

    def up(self):
        """Update / Create the cluster."""
        return self._run('up')

    def down(self):
        """Shutdown the cluster."""
        return self._run('down')

    def submit(self, runnable: Runnable):
        """Submit a job to the cluster."""
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            subprocess.run(f"ray submit {fp.name}")

    def head_node_ip(self) -> str:
        """Get the head node ip address"""
        return get_head_node_ip()

    def worker_node_ips(self) -> List[str]:
        """Get the worker node ip addresses."""
        return get_worker_node_ips()
