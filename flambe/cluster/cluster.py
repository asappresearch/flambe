import os
import copy
from typing import Optional, Dict, List, Tuple
from ruamel.yaml import YAML
import tempfile
import inspect
from datetime import datetime
from functools import partial
from ruamel.yaml import comments


from ray.autoscaler.commands import get_head_node_ip, get_worker_node_ips
from ray.autoscaler.commands import exec_cluster, create_or_update_cluster, rsync, teardown_cluster

import flambe
from flambe.logging import coloredlogs as cl
from flambe.const import FLAMBE_GLOBAL_FOLDER, PYTHON_VERSION
from flambe.compile import Registrable, YAMLLoadType, load_config_from_file
from flambe.compile import download_extensions, download_manager
from flambe.utils.path import is_dev_mode, get_flambe_repo_location, is_pip_installable
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
    """Get the current time.

    Returns
    -------
    str
        The current time, formatted as a string.

    """
    return datetime.now().strftime('%H:%M:%S')


def upload_files(local_files: Dict,
                 remote_files: Dict,
                 folder: Optional[str] = None) -> Tuple[Dict, Dict]:
    """Upload files to the cluster.

    Parameters
    ----------
    folder: str, optional
        The parent folder to upload the files to.

    Returns
    -------
    Dict[str, str]
        The updated file mount.
    Dict[str, str]
        The updated local files mapping.

    """
    files_dir = os.path.join(FLAMBE_GLOBAL_FOLDER, 'files')
    file_mounts: Dict[str, str] = dict()
    updated_files: Dict[str, str] = dict()
    updated_files.update(remote_files)
    for file_name, file in local_files.items():
        with download_manager(file, os.path.join(files_dir, file_name)) as path:
            target = os.path.join(folder, f'files/{file_name}')
            file_mounts[target] = path
            updated_files[file_name] = target
    return updated_files, file_mounts


def upload_flambe(folder: Optional[str] = None) -> Tuple[str, Dict]:
    """Set up Flambé on the cluster.

    Returns
    -------
    str
        New command to run.
    Dict[str, str]
        New file mounts.

    """
    file_mounts: Dict[str, str] = dict()
    if is_dev_mode():
        flambe_repo = get_flambe_repo_location()
        target = f'flambe_dev'
        if folder is not None:
            target = os.path.join(folder, target)
        file_mounts[target] = flambe_repo
        cmd = f'pip install -U -e {target}'
    else:
        cmd = f'pip install -U flambe=={flambe.__version__}'

    return cmd, file_mounts


def upload_extensions(extensions: Dict, folder: Optional[str] = None) -> Tuple[List, Dict, Dict]:
    """Set up extensions on the cluster.

    Parameters
    ----------
    extensions : Dict[str, str]
        [description]

    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        [description]

    """
    folder = folder if folder is not None else ''

    extensions_dir = os.path.join(FLAMBE_GLOBAL_FOLDER, 'extensions')
    if not os.path.exists(extensions_dir):
        os.makedirs(extensions_dir)

    extensions = download_extensions(extensions, extensions_dir)
    updated_extensions: Dict[str, str] = dict()
    file_mounts: Dict[str, str] = dict()
    commands = []
    for module, package in extensions.items():
        target = package
        cmd = ''
        if os.path.exists(package):
            package = os.path.join(package, '')
            target = os.path.join(folder, f'extensions/{module}')
            if not is_pip_installable(package):
                target = os.path.join(target, '')
                python_setup = f"from setuptools import setup, find_packages\
                \n\nsetup(name=flambe__{module}, version='0.0.0')"
                cmd += f'echo "{python_setup}" > {target}setup.py && '
            file_mounts[target] = package
        cmd += f"pip install -U {target}"
        updated_extensions[module] = target
        commands.append(cmd)

    return commands, file_mounts, updated_extensions


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
                 autoscaling_mode: str = 'default',
                 target_utilization_fraction: float = 0.8,
                 idle_timeout_minutes: int = 5,
                 ssh_user: str = 'ubuntu',
                 ssh_private_key: Optional[str] = None,
                 file_mounts: Optional[Dict[str, str]] = None,
                 setup_commands: Optional[List[str]] = None,
                 head_setup_commands: Optional[List[str]] = None,
                 worker_setup_commands: Optional[List[str]] = None,
                 head_start_ray_commands: Optional[List[str]] = None,
                 worker_start_ray_commands: Optional[List[str]] = None,
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
        autoscaling_mode: str
            One of: ['default', 'aggressive']. Whether or not to
            autoscale aggressively. If this is enabled, if at any point
            we would start more workers, we start at least enough
            to bring us to initial_workers.
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
        head_start_ray_commands : Optional[List[str]], optional
            A list of commands to start ray on the head node.
        worker_start_ray_commands : Optional[List[str]], optional
            A list of commands to start ray on worker nodes.
        extra: Dict, optional
            Extra arguments to the cluster conifg. See all available
            arguments on the Ray Autoscaler documentation.

        """
        # Upload Flambé and extensions from cluster config
        permanent_cmds, permanent_files = [], dict()
        permanent_cmds.append(f"conda create -y -q --name flambe python={PYTHON_VERSION}; echo ''")

        setup_commands = setup_commands or []
        setup_commands.append("pip install aiohttp psutil setproctitle grpcio")

        flambe_cmd, files = upload_flambe()
        setup_commands.append(flambe_cmd)
        permanent_files.update(files)

        env = flambe.get_env()
        if env.extensions:
            cmds, files, _ = upload_extensions(env.extensions)
            setup_commands.extend(cmds)
            permanent_files.update(files)

        # Set envrionment for all subsequent commands
        command_lists = [
            setup_commands,
            head_setup_commands,
            worker_setup_commands,
            head_start_ray_commands,
            worker_start_ray_commands
        ]
        for commands in filter(lambda x: x is not None, command_lists):
            for i, cmd in enumerate(commands):
                commands[i] = f"source activate flambe && {cmd}"

        permanent_cmds = permanent_cmds + setup_commands

        # Set permanent commands and file mounts
        file_mounts = file_mounts or dict()
        permanent_files.update(file_mounts)

        self.config = {
            'cluster_name': name,
            'initial_workers': initial_workers,
            'min_workers': min_workers,
            'max_workers': max_workers,
            'autoscaling_mode': autoscaling_mode,
            'target_utilization_fraction': target_utilization_fraction,
            'idle_timeout_minutes': idle_timeout_minutes,
            'auth': {
                'ssh_user': ssh_user,
                'ssh_private_key': ssh_private_key
            },
            'file_mounts': permanent_files,
            'setup_commands': permanent_cmds,
            'head_setup_commands': head_setup_commands or [],
            'worker_setup_commands': worker_setup_commands or [],
            'head_start_ray_commands': head_start_ray_commands or [],
            'worker_start_ray_commands': worker_start_ray_commands or [],
            'initialization_commands': []
        }

        # Add extra config
        for key, value in extra.items():
            if key in self.config:
                raise ValueError("Extra should not modify the permanent config.")
            else:
                self.config[key] = value

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

    def up(self, yes: bool = False, restart: bool = False):
        """Update / Create the cluster.

        Parameters
        ----------
        yes : bool, optional
            Whether to force confirm a change in the cluster.
        restart : bool, optional
            Whether to restart the ray services.

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            create_or_update_cluster(fp.name, None, None, not restart, False, yes, None)

        print(cl.GR(f"Cluster started successful."))

    def down(self, yes: bool = False, workers_only: bool = False, destroy: bool = False):
        """Teardown the cluster.

        Parameters
        ----------
        yes : bool, optional
            Tear the cluster down.
        workers_only : bool, optional
            Kill only worker nodes, by default False.
        destroy: bool, optional
            Destroy this cluster permanently.

        """
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            teardown_cluster(fp.name, yes, workers_only, None)

        print(cl.GR(f"Teardown successful."))

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
        print("\n", "-" * 50)
        print(cl.BL(f"\nCluster: {self.config['cluster_name']}"))
        print(f"Head IP: {self.head_node_ip()}")
        print(f"Worker IP's: {self.worker_node_ips()}\n")

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
               debug: bool = False,
               num_cpus: int = 1,
               num_gpus: int = 0):
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
        num_cpus : int, optional
            The number of CPU's to use for this job.
            Default ``1``.
        num_cpus : int, optional
            The number of GPU's to use for this job.
            Default ``0``.

        """
        print(cl.BL(f'[{time()}] Submitting Job: {name}.'))

        # Load environment
        env = load_env_from_config(runnable)
        if env is None:
            env = Environment()

        # Check running jobs for conflict
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            # Check if a job is currently running on the cluster
            try:
                with capture_ray_output(supress_all=True):
                    cmd = f"ps -C flambe"
                    exec_cluster(fp.name, cmd)
                running = True
            except:  # noqa: E722
                running = False
            restart = not running
            if running and not force:
                msg = "Another job is currently running.\n"
                msg += "The cluster only supports running a single virtual env at a time. "
                msg += "If you would like to run this job in the current virtual env, "
                msg += "use -f, --force at your own risks. Note that if your new job, "
                msg += "has different extensions or file mounts than the currently running job, "
                msg += "those will be replaced in the cluster configuration, which will impact "
                msg += "what gets installed and synced to workers created by autoscaling."
                print(cl.RE(msg))
                return

        # Prepare list of commands and file mounts
        file_mounts = {}
        setup_cmds = []
        head_cmds = []
        worker_cmds = []
        job_env_name = "current"

        # Create new directory, new tmux session, and virtual env
        cmd = f"tmux kill-session -t {name}; tmux new -d -s {name}"
        head_cmds.append(cmd)

        cmd = f"mkdir -p jobs/{name}/extensions && "
        if restart:
            cmd += f"conda remove -y -q --name {job_env_name} --all; "
        cmd += f"conda create --name {job_env_name} --clone flambe; echo ''"
        setup_cmds.append(cmd)

        # Run the next head node commands through tmux
        # and in a clean flambe virtual env
        tmux = lambda x: f'tmux send-keys -t {name}.0 "{x}" ENTER'  # noqa: E731
        activate = lambda x: f"source activate {job_env_name} && {x}"  # noqa: E731

        # Upload Flambé
        cmd, flambe_files = upload_flambe(folder=f"jobs/{name}")
        head_cmds.append(tmux(activate(cmd)))
        worker_cmds.append(activate(cmd))
        file_mounts.update(flambe_files)

        # Upload extensions
        commands, files, updated_exts = upload_extensions(env.extensions, folder=f"jobs/{name}")
        head_cmds.extend([tmux(activate(cmd)) for cmd in commands])
        worker_cmds.extend([activate(cmd) for cmd in commands])
        file_mounts.update(files)

        # Upload files
        mounts, updated_files = upload_files(
            env.local_files,
            env.remote_files,
            folder=f"jobs/{name}"
        )
        file_mounts.update(mounts)

        # Run Flambe
        env = env.clone(
            head_node_ip=self.head_node_ip(),
            worker_node_ips=self.worker_node_ips(),
            extensions=updated_exts,
            local_files=updated_files,
            remote_files=[],
            remote=True
        )

        # Rsync files and run setup commands
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as config_file:
            set_env_in_config(env, runnable, config_file)
            config_target = f"jobs/{name}/{os.path.basename(runnable)}"
            file_mounts[config_target] = config_file.name

            config: Dict = copy.deepcopy(self.config)
            config['file_mounts'].update(file_mounts)
            config['setup_commands'].extend(setup_cmds)
            config['head_setup_commands'].extend(head_cmds)
            config['worker_setup_commands'].extend(worker_cmds)

            # Must add source activate to every other command
            with tempfile.NamedTemporaryFile() as fp:
                yaml.dump(config, fp)
                with capture_ray_output(supress_all=True) as output:  # noqa:
                    create_or_update_cluster(fp.name, None, None, not restart, False, True, None)
                if env.debug:
                    print(output)

        # Run Flambe
        options = f' --output jobs/{name}'
        options += f' --num-cpus {num_cpus} --num-gpus {num_gpus}'
        options += ' -d' * int(debug) + ' -f' * int(force)
        cmd = f"flambe run {config_target}{options}; "
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(self.config, fp)
            with capture_ray_output(supress_all=True):
                exec_cluster(fp.name, tmux(activate(cmd)))

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
