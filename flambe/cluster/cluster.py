import os
from typing import Optional, Dict, List
from ruamel.yaml import YAML
import tempfile

from ray.autoscaler.commands import get_head_node_ip, get_worker_node_ips
from ray.autoscaler.commands import exec_cluster, create_or_update_cluster, rsync

from flambe.compile import RegisteredStatelessMap, load_extensions
from flambe.runner.utils import is_dev_mode, get_flambe_repo_location


class Cluster(RegisteredStatelessMap):
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
        config = {
            'cluster_name': name,
            'initial_workers': initial_workers,
            'min_workers': min_workers,
            'max_workers': max_workers,
            'target_utilization_fraction': target_utilization_fraction,
            'idle_timeout_minutes': idle_timeout_minutes,
            'auth': {
                'ssh_user': ssh_user,
                'ssh_private_key': ssh_private_key
            },
            'file_mounts': file_mounts or {},
            'setup_commands': setup_commands or [],
            'head_setup_commands': head_setup_commands or [],
            'worker_setup_commands': worker_setup_commands or [],
            'initialization_commands': []
        }

        self.config = config

    def up(self,
           min_workers: Optional[int] = None,
           max_workers: Optional[int] = None,
           yes: bool = False):
        """Update / Create the cluster.

        Parameters
        ----------
        min_workers : Optional[int], optional
            [description], by default None
        max_workers : Optional[int], optional
            [description], by default None
        yes : bool, optional
            [description], by default False

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            create_or_update_cluster(fp.name, min_workers, max_workers, True, False, yes, None)

    def down(self):
        """Teardown the cluster."""
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            teardown_cluster(fp.name, yes, workers_only, cluster_name)

    def rsync_up(self, source: str, target: str):
        """[summary]

        Parameters
        ----------
        source : str
            [description]
        target : str
            [description]
        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            rsync(fp.name, source, target, down=False)

    def rsync_down(self, source: str, target: str):
        """[summary]

        Parameters
        ----------
        source : str
            [description]
        target : str
            [description]
        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            rsync(fp.name, source, target, down=True)

    def attach(self, name: str, new: bool = False):
        """Attach to a tmux session.

        Arguments:
            new: whether to force a new tmux session

        """
        if new:
            cmd = f"tmux new -s {name}"
        else:
            cmd = f"tmux attach -t {name} || tmux new -s {name}"

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            exec_cluster(fp.name, cmd, False, False, False, False, False, None, None)

    def list(self):
        """Attach to a tmux session.

        Arguments:
            new: whether to force a new tmux session

        """
        cmd = f"tmux ls"

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            exec_cluster(fp.name, cmd, False, False, False, False, False, None, None)

    def kill(self, name: str):
        """Attach to a tmux session.

        Arguments:
            new: whether to force a new tmux session

        """
        cmd = f"tmux kill-session -t {name}"

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            exec_cluster(fp.name, cmd, False, False, False, False, False, None, None)

    def submit(self,
               runnable: str,
               name: str,
               force: bool = False,
               debug: bool = False):
        """Submit a job to the cluster.

        Parameters
        ----------
        runnable : str
            [description]
        name : str
            [description]
        force : bool, optional
            [description], by default False
        debug : bool, optional
            [description], by default False

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)

            # Create new directory, tmux session, and virtual env
            cmd = f"mkdir ~/{name} && tmux new -s {name}"
            cmd += f' && conda create -y -q --name {name} && source activate {name}'
            exec_cluster(fp.name, cmd, False, False, False, False, False, None, [])
            tmux_cmd = lambda x: f'tmux send-keys -t {name}.0 "{x}" ENTER'  # noqa: E731

            # Upload and install flambe
            if is_dev_mode():
                flambe_repo = get_flambe_repo_location()
                target = f'~/{name}/flambe'
                rsync(fp.name, flambe_repo, target, None, down=False)

                cmd = f'pip install -U {target}'
                exec_cluster(fp.name, tmux_cmd(cmd), False, False, False, False, False, None, [])
            else:
                cmd = f'pip install -U flambe'
                exec_cluster(fp.name, tmux_cmd(cmd), False, False, False, False, False, None, [])

            # Upload and install extensions
            extensions = load_extensions(runnable)
            for module, package in extensions.items():
                target = package
                if os.path.exists(package):
                    target = f'~/{name}/extensions/{os.path.basename(package)}'
                    rsync(fp.name, package, target, None, down=False)

                cmd = f'pip install -U {target}'
                exec_cluster(fp.name, tmux_cmd(cmd), False, False, False, False, False, None, [])

            # Run Flambe
            target = os.path.join(f"~/{name}", os.path.basename(runnable))
            rsync(fp.name, runnable, target, None, down=False)
            cmd = f'flambe run {target}'
            exec_cluster(fp.name, tmux_cmd(cmd), False, False, False, False, False, None, [])

    def head_node_ip(self) -> str:
        """Get the head node ip address"""
        return get_head_node_ip()

    def worker_node_ips(self) -> List[str]:
        """Get the worker node ip addresses."""
        return get_worker_node_ips()
