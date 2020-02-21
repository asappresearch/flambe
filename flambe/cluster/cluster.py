import os
import copy
from typing import Optional, Dict, List
from ruamel.yaml import YAML
import tempfile
import inspect
from datetime import datetime
from functools import partial
from ruamel.yaml import comments


from ray.autoscaler.commands import get_head_node_ip, get_worker_node_ips
from ray.autoscaler.commands import exec_cluster, create_or_update_cluster, rsync, teardown_cluster

from flambe.logging import coloredlogs as cl
from flambe.const import FLAMBE_GLOBAL_FOLDER, PYTHON_VERSION
from flambe.compile import Registrable, YAMLLoadType, load_config_from_file
from flambe.compile import download_extensions, download_manager
from flambe.utils.path import is_dev_mode, get_flambe_repo_location
from flambe.utils.ray import capture_ray_output
from flambe.runner.environment import Environment, load_env_from_config, set_env_in_config


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


def time() -> str:
    """Get the current time."""
    return datetime.now().strftime('%H:%M:%S')


class Cluster(Registrable):
    """Base cluster implementation."""

    # TODO: clean this
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        params = inspect.signature(cls.__init__).parameters
        for arg, name in zip(args, list(params.keys())[1:]):  # ignore self
            kwargs[name] = arg
        params = {
            k: v.default for k, v in params.items(
            ) if k != 'self' and v.default != inspect._empty
        }
        params.update(kwargs)
        params = comments.CommentedMap(params)
        name = cls.__name__ if 'flambe.' in cls.__module__ else f"{cls.__module__}.{cls.__name__}"
        params.yaml_set_tag(f"!{name}")
        instance._saved_arguments = params
        return instance

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
                 extra: Optional[Dict] = None) -> None:
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
        extra: Dict, optional
            Extra arguments to the cluster conifg. See all available
            arguments on the Ray Autoscaler documentation.

        """
        self.config = {
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
        if extra:
            self.config.update(extra)

    def clone(self, **kwargs) -> 'Cluster':
        """Clone the cluster, updated with the provided arguments.

        Parameters
        ----------
        **kwargs: Dict
            Arguments to override.

        Returns
        -------
        Cluster
            The new updated cluster object.

        """
        arguments = copy.deepcopy(self._saved_arguments)  # type: ignore
        arguments.update(kwargs)
        return self.__class__(**arguments)  # type: ignore

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.KWARGS

    def up(self, yes: bool = False):
        """Update / Create the cluster.

        Parameters
        ----------
        yes : bool, optional
            Whether to force confirm a change in the cluster.

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            create_or_update_cluster(fp.name, None, None, True, False, yes, None)

    def down(self, yes: bool = False, workers_only: bool = False):
        """Teardown the cluster.

        Parameters
        ----------
        yes : bool, optional
            Tear the cluster down.
        workers_only : bool, optional
            Kill only worker nodes, by default False.

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            teardown_cluster(fp.name, yes, workers_only, None)

    def rsync_up(self, source: str, target: str):
        """Rsync the local source to the target on the cluster.

        Parameters
        ----------
        source : str
            The source folder on the local machine.
        target : str
            The target folder on the cluster.

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            rsync(fp.name, source, target, None, down=False)

    def rsync_down(self, source: str, target: str):
        """Rsync the source from the cluster to the local target.

        Parameters
        ----------
        source : str
            The source folder on the cluster.
        target : str
            The target folder on the local machine.

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            rsync(fp.name, source, target, None, down=True)

    def attach(self, name: str, new: bool = False):
        """Attach onto a running job (i.e tmux session).

        Parameters
        ----------
        name: str
           The name of the job to attach.
        new: bool
            Whether to create a new tmux sessions. Note that
            if there is already a session with this name, new
            has no effect. Default ``False``.

        """
        cmd = f"tmux attach -t {name}"
        if new:
            cmd += f" || tmux new-session -s {name}"

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            exec_cluster(fp.name, cmd)

        # Seems to help with terminal sometime hanging after detach
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

    def kill(self, name: str):
        """Kill a running jog (i.e tmux session).

        Parameters
        ----------
        name: str
            The name of the job to kill.

        """
        cmd = f"tmux kill-session -t {name}"

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            with capture_ray_output():
                exec_cluster(fp.name, cmd)

        print(cl.GR(f"Job {name} killed."))

    def clean(self, name: str):
        """Clean the artifacts of a job.

        Parameters
        ----------
        name: str
            The name of the job to clean.

        """
        cmd = f"rm -rf $HOME/jobs/{name}"

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            with capture_ray_output():
                exec_cluster(fp.name, cmd)

        print(cl.GR(f"Job {name} cleaned."))

    def list(self):
        """List all currently running jobs (i.e tmux sessions)."""
        cmd = f'tmux ls || echo "No tmux session running"'

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            with capture_ray_output():
                exec_cluster(fp.name, cmd)

    def exec(self, command: str, port_forward: Optional[int] = None):
        """Run a command on the cluster.

        Parameters
        ----------
        command: str
            The command to run.
        port_forward: int, optional
            An optional port to use for port fowarding.

        """
        port_forward = port_forward if port_forward is not None else []

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            with capture_ray_output():
                exec_cluster(fp.name, command, port_forward=port_forward)

    def launch_site(self, name: str = '', port: int = 4444):
        """Launch the report website.

        Parameters
        ----------
        name: str, optional
            If given, restricts the report to a single job.
        port: int
            The port to use in launching the site.

        """
        cmd = f'pip install tensorboard > /dev/null && \
            tensorboard --logdir=$HOME/jobs/{name} --port={port}'

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            print(cl.BL(f"Preparing site at: http://localhost:{port}"))
            with capture_ray_output():
                exec_cluster(fp.name, cmd, port_forward=port)

    def submit(self,
               runnable: str,
               name: str,
               force: bool = False,
               debug: bool = False):
        """Submit a job to the cluster.

        Parameters
        ----------
        runnable : str
            The runnable config to submit.
        name : str
            A name for the job.
        force : bool, optional
            Whether to override a previous job of the same name.
            Default ``False``.
        debug : bool, optional
            Whether to run in debug mode.
            Default ``False``.

        """
        # Load environment
        env = load_env_from_config(runnable)
        if env is None:
            env = Environment()

        # Turn off output from ray
        print(cl.BL(f'[{time()}] Submitting Job: {name}.'))

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            # Check if a job is currently running
            try:
                with capture_ray_output(supress_all=True):
                    cmd = f"tmux has-session -t {name}"
                    exec_cluster(fp.name, cmd)
                running = True
            except:  # noqa: E722
                running = False
            if running and not force:
                print(cl.RE(f"Job {name} currently running. Use -f, --force to override."))
                return

        # Create new directory, new tmux session, and virtual env
        head_setup_commands = []
        worker_setup_commands = []
        file_mounts = {}

        cmd = f"tmux kill-session -t {name}; tmux new -d -s {name}"
        head_setup_commands.append(cmd)

        tmux = lambda x: f'tmux send-keys -t {name}.0 "{x}" ENTER'  # noqa: E731
        cmd = f"mkdir -p $HOME/jobs/{name}/extensions && \
            conda create -y -q --name {name} python={PYTHON_VERSION}; echo ''"
        head_setup_commands.append(tmux(cmd))
        worker_setup_commands.append(cmd)

        # Upload and install flambe
        activate = f'source activate {name}'
        if is_dev_mode():
            flambe_repo = get_flambe_repo_location()
            target = f'~/jobs/{name}/flambe'
            file_mounts[target] = flambe_repo

            target = f'~/jobs/{name}/flambe'
            cmd = f'{activate} && pip install -U -e {target}'
        else:
            cmd = f'{activate} && pip install -U flambe'

        head_setup_commands.append(tmux(cmd))
        worker_setup_commands.append(cmd)

        # Upload and install extensions
        extensions = env.extensions
        extensions_dir = os.path.join(FLAMBE_GLOBAL_FOLDER, 'extensions')
        if not os.path.exists(extensions_dir):
            os.makedirs(extensions_dir)

        extensions = download_extensions(extensions, extensions_dir)
        updated_extensions: Dict[str, str] = dict()
        for module, package in extensions.items():
            target = package
            if os.path.exists(package):
                package = os.path.join(package, '')
                target = f'~/jobs/{name}/extensions/{module}'
                file_mounts[target] = package
                target = f'~/jobs/{name}/extensions/{module}'

            updated_extensions[module] = target
            cmd = f'pip install -U {target}'
            head_setup_commands.append(tmux(cmd))
            worker_setup_commands.append(cmd)

        # Upload files
        files_dir = os.path.join(FLAMBE_GLOBAL_FOLDER, 'files')
        updated_files: Dict[str, str] = dict()
        updated_files.update(env.remote_files)
        for file_name, file in env.local_files.items():
            with download_manager(file, os.path.join(files_dir, file_name)) as path:
                target = f'~/jobs/{name}/files/{file_name}'
                file_mounts[target] = path
                updated_files[file_name] = target

        # Run Flambe
        env = env.clone(
            output_path=f"~/jobs/{name}",
            head_node_ip=self.head_node_ip(),
            worker_node_ips=self.worker_node_ips(),
            extensions=updated_extensions,
            local_files=updated_files,
            remote_files=[],
            remote=True
        )

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as config_file:
            set_env_in_config(env, runnable, config_file)
            config_target = f"~/jobs/{name}/{os.path.basename(runnable)}"
            file_mounts[config_target] = config_file.name

            # Rsync files and run setup commands
            config: Dict = copy.deepcopy(self.config)
            config['head_setup_commands'].extend(head_setup_commands)
            config['worker_setup_commands'].extend(worker_setup_commands)
            config['file_mounts'].update(file_mounts)
            with tempfile.NamedTemporaryFile() as fp:
                yaml.dump(config, fp)
                with capture_ray_output(supress_all=True) as output:  # noqa:
                    create_or_update_cluster(fp.name, None, None, True, False, True, None)
                if env.debug:
                    print(output)

        # Run Flambe
        cmd = f'flambe run {config_target}'
        cmd += ' -d' * int(debug) + ' -f' * int(force)
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            with capture_ray_output(supress_all=True):
                exec_cluster(fp.name, tmux(cmd))

        print(cl.GR(f'[{time()}] Job submitted successfully.\n'))

    def head_node_ip(self) -> str:
        """Get the head node ip address.

        Returns
        -------
        str
            The head node IP address.

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            node_ip = get_head_node_ip(fp.name, None)
        return node_ip

    def worker_node_ips(self) -> List[str]:
        """Get the worker node ip addresses.

        Returns
        -------
        List[str]
            The worker nodes IP addresses.

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            node_ips = get_worker_node_ips(fp.name, None)
        return node_ips


def load_cluster_config(path: str) -> Cluster:
    """Load a Cluster obejct from the given config.

    Parameters
    ----------
    path : str
        A path to the cluster config.

    Returns
    -------
    Cluster
        The loaded cluster object

    """
    configs = list(load_config_from_file(path))
    return configs[-1]
