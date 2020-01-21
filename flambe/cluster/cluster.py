import os
from typing import Optional, Dict, List
from ruamel.yaml import YAML
import tempfile
import subprocess
from datetime import datetime
from functools import partial

from ray.autoscaler.commands import get_head_node_ip, get_worker_node_ips
from ray.autoscaler.commands import exec_cluster, create_or_update_cluster, rsync, teardown_cluster
from ray.autoscaler.updater import SSHCommandRunner

from flambe.logging import coloredlogs as cl
from flambe.const import FLAMBE_GLOBAL_FOLDER
from flambe.compile import RegisteredStatelessMap
from flambe.compile import load_extensions_from_file
from flambe.compile.extensions import download_extensions
from flambe.runner.utils import is_dev_mode, get_flambe_repo_location


exec_cluster = partial(
    exec_cluster,
    docker=False,
    screen=False,
    tmux=False,
    stop=False,
    start=False,
    override_cluster_name=None,
    port_forward=[]
)


def supress():
    """Supress the messages coming from the ssh command."""
    def supress_ssh(func):
        def wrapper(self, connect_timeout):
            return func(self, connect_timeout) + ['-o', "LogLevel=QUIET"]
        return wrapper

    def supress_rsync(func):
        def wrapper(args, *, stdin=None, stdout=None, stderr=None,
                    shell=False, cwd=None, timeout=None):
            if args[0] == 'rsync':
                stdout = subprocess.DEVNULL
            func(args, stdin=None, stdout=stdout, stderr=None,
                 shell=False, cwd=None, timeout=None)
        return wrapper

    SSHCommandRunner.get_default_ssh_options = supress_ssh(
        SSHCommandRunner.get_default_ssh_options
    )
    subprocess.check_call = supress_rsync(subprocess.check_call)


def time() -> str:
    """Get the current time."""
    return datetime.now().strftime('%H:%M:%S')


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
                 worker_setup_commands: Optional[List[str]] = None,
                 custom_config: Optional[Dict] = None) -> None:
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
        custom_config: str, optional

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
           yes: bool = True):
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

    def down(self, yes: bool = True, workers_only: bool = False):
        """Teardown the cluster.

        Parameters
        ----------
        yes : bool, optional
            [description], by default True
        workers_only : bool, optional
            [description], by default False

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            teardown_cluster(fp.name, yes, workers_only, None)

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
            rsync(fp.name, source, target, None, down=False)

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
            rsync(fp.name, source, target, None, down=True)

    def attach(self, name: Optional[str] = None):
        """Attach to a tmux session.

        Parameters
        ----------
        name: str, optional
           The name of the job to attached to. If none give,
           creates a new tmux session. Default ``None``.

        """
        supress()

        if name is not None:
            cmd = f"tmux attach -t {name}"
        else:
            cmd = f"tmux new"

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            exec_cluster(fp.name, cmd)

        # Seems to help with terminal sometime hanging after detach
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

    def list(self):
        """Attach to a tmux session.

        Arguments:
            new: whether to force a new tmux session

        """
        cmd = f'tmux ls || echo "No tmux session running"'

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            exec_cluster(fp.name, cmd)

    def kill(self, name: str):
        """Attach to a tmux session.

        Arguments:
            new: whether to force a new tmux session

        """
        cmd = f"tmux kill-session -t {name}"

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            exec_cluster(fp.name, cmd)

    def launch_site(self, port, name):
        """Attach to a tmux session.

        Arguments:
            new: whether to force a new tmux session

        """
        cmd = f'pip install tensorboard > /dev/null && \
            tensorboard --logdir=$HOME/jobs/{name} --port={port}'

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            try:
                exec_cluster(fp.name, cmd, port_forward=port)
            except KeyboardInterrupt:
                pass

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
        # Turn off output from ray
        supress()
        print(cl.BL(f'[{time()}] Submitting Job: {name}.'))

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)

            # Check if a job is currently running
            try:
                cmd = f"tmux has-session -t {name}"
                exec_cluster(fp.name, cmd)
                running = True
            except:  # noqa: E722
                running = False
            if running and not force:
                raise ValueError(f"Job {name} currently running. Use -f, --force to override.")

            # Create new directory, new tmux session, and virtual env
            cmd = f"tmux kill-session -t {name}; tmux new -d -s {name}"
            exec_cluster(fp.name, cmd)

            tmux = lambda x: f'tmux send-keys -t {name}.0 "{x}" ENTER'  # noqa: E731
            cmd = f"mkdir -p $HOME/jobs/{name}/extensions && \
                conda create -y -q --name {name}; echo ''"
            exec_cluster(fp.name, tmux(cmd))

            # Upload and install flambe
            cmd = f'source activate {name}'
            if is_dev_mode():
                flambe_repo = get_flambe_repo_location()
                target = f'~/jobs/{name}'
                rsync(fp.name, flambe_repo, target, None, down=False)

                cmd += f' && pip install -U -e {target}/flambe'
                exec_cluster(fp.name, tmux(cmd))
            else:
                cmd += f' && pip install -U flambe'
                exec_cluster(fp.name, tmux(cmd))

            # Upload and install extensions
            extensions_dir = os.path.join(FLAMBE_GLOBAL_FOLDER, 'extensions')
            if not os.path.exists(extensions_dir):
                os.makedirs(extensions_dir)
            extensions = load_extensions_from_file(runnable)
            extensions = download_extensions(extensions, extensions_dir)
            for module, package in extensions.items():
                target = package
                if os.path.exists(package):
                    target = f'$HOME/jobs/{name}/extensions'
                    rsync(fp.name, package, target, None, down=False)
                    target = f'{target}/{os.path.basename(package)}'

                cmd = f'pip install -U {target}'
                exec_cluster(fp.name, tmux(cmd))

            # Run Flambe
            env = {
                'output_path': f"~/jobs/{name}",
                'remote': True,
                'head_node_ip': self.head_node_ip()
            }
            yaml = YAML()
            with tempfile.NamedTemporaryFile() as env_file:
                yaml.dump(env, env_file)
                env_target = f"$HOME/jobs/{name}/env.yaml"
                rsync(fp.name, env_file.name, env_target, None, down=False)

            # Run Flambe
            target = f"$HOME/jobs/{name}/{os.path.basename(runnable)}"
            rsync(fp.name, runnable, target, None, down=False)
            cmd = f'flambe run {target} --env {env_target}'
            cmd += ' -d' * int(debug) + ' -f' * int(force)
            exec_cluster(fp.name, tmux(cmd))
            print(cl.GR(f'[{time()}] Job submitted successfully.\n'))

    def head_node_ip(self) -> str:
        """Get the head node ip address"""
        supress()
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            node_ip = get_head_node_ip(fp.name, None)
        return node_ip

    def worker_node_ips(self) -> List[str]:
        """Get the worker node ip addresses."""
        supress()
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            node_ips = get_worker_node_ips(fp.name, None)
        return node_ips
