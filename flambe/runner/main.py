import logging
import click
import os
import sys
import shutil
import traceback

import torch

import flambe
from flambe.const import FLAMBE_GLOBAL_FOLDER, ASCII_LOGO, ASCII_LOGO_DEV
from flambe.const import FLAMBE_CLUSTER_DEFAULT_FOLDER, FLAMBE_CLUSTER_DEFAULT_CONFIG
from flambe.logging import coloredlogs as cl
from flambe.utils.path import is_dev_mode, get_flambe_repo_location
from flambe.compile.yaml import dump_one_config
from flambe.compile.downloader import download_manager
from flambe.compile.extensions import is_package, is_installed_module
from flambe.runner.environment import load_env_from_config
from flambe.runner.protocol import load_runnable_from_config
from flambe.cluster.cluster import load_cluster_config


logging.getLogger('tensorflow').disabled = True


def load_cluster_config_helper(name=None, return_name=False):
    """Check if a single cluster exists or if a name is required."""
    if name is None:
        files = os.listdir(FLAMBE_CLUSTER_DEFAULT_FOLDER)
        names = [f.replace('.yaml', '') for f in files if '.yaml' in f]
        if len(names) == 1:
            name = names[0]
        elif len(names) == 0:
            print(cl.RE(f"There are no clusters at {FLAMBE_CLUSTER_DEFAULT_FOLDER}."))
            sys.exit()
        else:
            print(cl.RE(f"No name was provided, but multiple clusters exist: {names}"))
            sys.exit()
    cluster_path = os.path.join(FLAMBE_CLUSTER_DEFAULT_FOLDER, f"{name}.yaml")
    cluster = load_cluster_config(cluster_path)
    if return_name:
        return cluster, name
    return cluster


@click.group()
def cli():
    pass


# ----------------- flambe up ------------------ #
@click.command()
@click.argument('name', type=str, required=True)
@click.option('-y', '--yes', is_flag=True, default=False,
              help='Run without confirmation.')
@click.option('--create', is_flag=True, default=False,
              help='Create a new cluster.')
@click.option('--config', type=str, default=FLAMBE_CLUSTER_DEFAULT_CONFIG,
              help="Cluster template config.")
@click.option('--min-workers', type=int, default=None,
              help="Required name for a new cluster.")
@click.option('--max-workers', type=int, default=None,
              help="Optional max number of workers.")
def up(name, yes, create, config, min_workers, max_workers):
    """Launch / update the cluster."""
    if not os.path.exists(FLAMBE_CLUSTER_DEFAULT_FOLDER):
        os.makedirs(FLAMBE_CLUSTER_DEFAULT_FOLDER)
    cluster_path = os.path.join(FLAMBE_CLUSTER_DEFAULT_FOLDER, f"{name}.yaml")

    # Check update or install
    if create and os.path.exists(cluster_path):
        print(cl.RE(f"Cluster {name} already exists."))
        return
    elif create and not os.path.exists(config):
        print(cl.RE(f"Config {config} does not exist."))
        return
    elif not create and not os.path.exists(cluster_path):
        print(cl.RE(f"Cluster {name} does not exist. Did you mean to use --create?"))
        return

    # Update kwargs
    kwargs = dict(name=name)
    if min_workers is not None:
        kwargs['min_workers'] = min_workers
    if max_workers is not None:
        kwargs['max_workers'] = max_workers

    # Run update
    if create:
        cluster = load_cluster_config(config)
    else:
        cluster = load_cluster_config(cluster_path)

    cluster = cluster.clone(**kwargs)
    cluster.up(yes=yes)
    with open(cluster_path, 'w') as f:
        dump_one_config(cluster, f)


# ----------------- flambe down ------------------ #
@click.command()
@click.argument('name', type=str, required=True)
@click.option('-y', '--yes', is_flag=True, default=False,
              help='Run without confirmation.')
@click.option('--workers-only', is_flag=True, default=False,
              help='Only teardown the worker nodes.')
@click.option('--destroy', is_flag=True, default=False,
              help='Destroy this cluster permanently.')
def down(name, yes, workers_only, terminate, destroy):
    """Take down the cluster, optionally destroy it permanently."""
    cluster = load_cluster_config_helper(name)
    cluster.down(yes, workers_only, terminate)

    if destroy:
        cluster_path = os.path.join(FLAMBE_CLUSTER_DEFAULT_FOLDER, f"{name}.yaml")
        os.remove(cluster_path)


# ----------------- flambe rsync up ------------------ #
@click.command()
@click.argument('source', type=str, required=True)
@click.argument('target', type=str, required=True)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
def rsync_up(source, target, cluster):
    """Upload files to the cluster."""
    cluster = load_cluster_config_helper(cluster)
    cluster.rsync_up(source, target)


# ----------------- flambe rsync down ------------------ #
@click.command()
@click.argument('source', type=str, required=True)
@click.argument('target', type=str, required=True)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
def rsync_down(source, target, cluster):
    """Download files from the cluster."""
    cluster = load_cluster_config_helper(cluster)
    cluster.rsync_down(source, target)


# ----------------- flambe list ------------------ #
@click.command()
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('-a', '--all-clusters', is_flag=True, default=False,
              help="List jobs on all clusters.")
def list_cmd(cluster, all_clusters):
    """List the jobs (i.e tmux sessions) running on the cluster."""
    logging.disable(logging.INFO)

    if cluster is not None:
        cluster_obj = load_cluster_config_helper(cluster)
        cluster_obj.list()
    elif not all_clusters:
        cluster_obj, cluster = load_cluster_config_helper(return_name=True)
        cluster_obj.list()
    else:
        files = os.listdir(FLAMBE_CLUSTER_DEFAULT_FOLDER)
        clusters = [f.replace('.yaml', '') for f in files if '.yaml' in f]
        if len(clusters) == 0:
            print("No clusters to inspect.")
        else:
            for cluster in clusters:
                cluster_obj = load_cluster_config_helper(cluster)
                cluster_obj.list()
            print("\n", "-" * 50, "\n")


# ----------------- flambe exec ------------------ #
@click.command()
@click.argument('command', type=str)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('-p', '--port-forward', type=int, default=None,
              help='Port forwarding')
def exec_cmd(command, port_forward, cluster):
    """Execute a command on the cluster head node."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config_helper(cluster)
    cluster.exec(command=command, port_forward=port_forward)


# ----------------- flambe attach ------------------ #
@click.command()
@click.argument('name', required=False, type=str, default=None)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('-n', '--new', is_flag=True, default=False,
              help="Whether to create a new session.")
def attach(name, cluster, new):
    """Attach to a running job (i.e tmux session) on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config_helper(cluster)
    cluster.attach(name, new=new)


# ----------------- flambe kill ------------------ #
@click.command()
@click.argument('name', type=str)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('--clean', is_flag=True, default=False,
              help='Clean the artifacts of the job.')
def kill(name, cluster, clean):
    """Kill a job (i.e tmux session) running on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config_helper(cluster)
    cluster.kill(name=name)
    if clean:
        cluster.clean(name=name)


# ----------------- flambe clean ------------------ #
@click.command()
@click.argument('name', type=str)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
def clean(name, cluster):
    """Clean the artifacts of a job on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config_helper(cluster)
    cluster.clean(name=name)


# ----------------- flambe submit ------------------ #
@click.command()
@click.argument('config', type=str, required=True)
@click.argument('name', type=str, required=True)
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('-f', '--force', is_flag=True, default=False,
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-d', '--debug', is_flag=True, default=False,
              help='Enable debug mode. Each runnable specifies the debug behavior. \
                    For example for an Pipeline, Ray will run in a single thread \
                    allowing user breakpoints')
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Verbose console output')
@click.option('-a', '--attach', is_flag=True, default=False,
              help='Attach after submitting the job.')
def submit(config, name, cluster, force, debug, verbose, attach):
    """Submit a job to the cluster, as a YAML config."""
    if debug:
        logging.disable(logging.INFO)
    else:
        logging.disable(logging.ERROR)
    if is_dev_mode():
        print(cl.RA(ASCII_LOGO_DEV))
        print(cl.BL(f"Location: {get_flambe_repo_location()}\n"))
    else:
        print(cl.RA(ASCII_LOGO))
        print(cl.BL(f"VERSION: {flambe.__version__}\n"))

    cluster = load_cluster_config_helper(cluster)
    cluster.submit(config, name, force=force, debug=debug)
    if attach:
        cluster.attach(name)


# ----------------- flambe site ------------------ #
@click.command()
@click.argument('name', type=str, required=False, default='')
@click.option('-c', '--cluster', type=str, default=None,
              help="Cluster name.")
@click.option('-p', '--port', type=int, default=49558,
              help='Port on which the site will be running.')
def site(name, cluster, port):
    """Launch a Web UI to monitor the activity on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_cluster_config_helper(cluster)
    try:
        cluster.launch_site(port=port, name=name)
    except KeyboardInterrupt:
        logging.disable(logging.ERROR)


# ----------------- flambe run ------------------ #
@click.command()
@click.argument('config', type=str, required=True)
@click.option('-o', '--output', default='./',
              help='An output directory. A folder named \
              `flambe_output` will be created there.')
@click.option('-f', '--force', is_flag=True, default=False,
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-d', '--debug', is_flag=True,
              help='Enable debug mode. Each runnable specifies the debug behavior. \
                    For example for an Pipeline, Ray will run in a single thread \
                    allowing user breakpoints')
def run(config, output, force, debug):
    """Execute a runnable config."""
    # Load environment
    env = load_env_from_config(config)
    if not env:
        env = flambe.get_env()

    # Check if previous job exists
    output = os.path.join(os.path.expanduser(output), 'flambe_output')
    if os.path.exists(output):
        if force:
            shutil.rmtree(output)
        else:
            print(cl.RE(f"{output} already exists. Use -f, --force to override."))
            return

    os.makedirs(output)

    # torch.multiprocessing exists, ignore mypy
    # TODO: investigate if this is actually needed
    torch.multiprocessing.set_start_method('fork', force=True)  # type: ignore

    # Check if dev mode
    if is_dev_mode():
        print(cl.RA(ASCII_LOGO_DEV))
        print(cl.BL(f"Location: {get_flambe_repo_location()}\n"))
    else:
        print(cl.RA(ASCII_LOGO))
        print(cl.BL(f"VERSION: {flambe.__version__}\n"))

    # Check if debug
    if debug:
        print(cl.YE(f"Debug mode activated\n"))

    # Check that all extensions are importable
    message = "Module ({}) from package ({}) is not installed."
    for module, package in env.extensions.items():
        package = os.path.expanduser(package)
        if not is_installed_module(module):
            # Check if the package exsists locally
            if os.path.exists(package):
                if is_package(package):
                    # Package exsists locally but is not installed
                    print(message.format(module, package) + " Attempting to add to path.")
                # Try to add to the python path
                sys.path.append(package)
            else:
                raise ValueError(message.format(module, package))

    # Download files
    files_dir = os.path.join(FLAMBE_GLOBAL_FOLDER, 'files')
    updated_files: Dict[str, str] = dict()
    for name, file in env.local_files.items():
        with download_manager(file, os.path.join(files_dir, name)) as path:
            updated_files[name] = path

    try:
        # Execute runnable
        flambe.set_env(
            output_path=output,
            debug=debug,
            local_files=updated_files
        )

        runnable = load_runnable_from_config(config)
        _continue = True
        while _continue:
            _continue = runnable.run()
        print(cl.GR("------------------- Done -------------------"))
    except KeyboardInterrupt:
        print(cl.RE("---- Exiting early (Keyboard Interrupt) ----"))
    except Exception:
        print(traceback.format_exc())
        print(cl.RE("------------------- Error -------------------"))


if __name__ == '__main__':
    cli.add_command(up)
    cli.add_command(down)
    cli.add_command(kill)
    cli.add_command(clean)
    cli.add_command(list_cmd, name='ls')
    cli.add_command(exec_cmd, name='exec')
    cli.add_command(attach)
    cli.add_command(run)
    cli.add_command(submit)
    cli.add_command(site)
    cli.add_command(rsync_up, name='rsync-up')
    cli.add_command(rsync_down, name='rsync-down')
    cli(prog_name='flambe')
