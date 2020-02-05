import logging
import click
import os
import shutil
import traceback

import torch

import flambe
from flambe.const import FLAMBE_CLUSTER_DEFAULT
from flambe.logo import ASCII_LOGO, ASCII_LOGO_DEV
from flambe.logging import coloredlogs as cl
from flambe.runner.utils import is_dev_mode, get_flambe_repo_location
from flambe.compile.yaml import load_config_from_file
from flambe.runner.runnable import Environment


@click.group()
def cli():
    pass


# ----------------- flambe up ------------------ #
@click.command()
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
def up(cluster):
    """Launch / update the cluster based on the given config"""
    cluster = load_config_from_file(cluster)
    cluster.up()


# ----------------- flambe down ------------------ #
@click.command()
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
def down(cluster):
    """Teardown the cluster."""
    cluster = load_config_from_file(cluster)
    cluster.down()


# ----------------- flambe rsync up ------------------ #
@click.command()
@click.argument('source', type=str, required=True)
@click.argument('target', type=str, required=True)
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
def rsync_up(source, target, cluster):
    """Upload files to the cluster."""
    cluster = load_config_from_file(cluster)
    cluster.rsync_up(source, target)


# ----------------- flambe rsync down ------------------ #
@click.command()
@click.argument('source', type=str, required=True)
@click.argument('target', type=str, required=True)
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
def rsync_down(source, target, cluster):
    """Download files from the cluster."""
    cluster = load_config_from_file(cluster)
    cluster.rsync_down(source, target)


# ----------------- flambe list ------------------ #
@click.command()
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
def list_cmd(cluster):
    """List the jobs (i.e tmux sessions) running on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_config_from_file(cluster)
    cluster.list()


# ----------------- flambe exec ------------------ #
@click.command()
@click.argument('command', type=str)
@click.option('-p', '--port-forward', type=int, default=None,
              help='Port in which the site will be running url')
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
def exec_cmd(command, port_forward, cluster):
    """Execute a command on the cluster head node."""
    logging.disable(logging.INFO)
    cluster = load_config_from_file(cluster)
    cluster.exec(command=command, port_forward=port_forward)


# ----------------- flambe attach ------------------ #
@click.command()
@click.argument('name', required=False, type=str, default=None)
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
def attach(name, cluster):
    """Attach to a running job (i.e tmux session) on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_config_from_file(cluster)
    cluster.attach(name)


# ----------------- flambe kill ------------------ #
@click.command()
@click.argument('name', type=str)
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
def kill(name, cluster):
    """Kill a job (i.e tmux session) running on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_config_from_file(cluster)
    cluster.kill(name=name)


# ----------------- flambe run ------------------ #
@click.command()
@click.argument('runnable', type=str, required=True)
@click.option('-o', '--output', default='./',
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-f', '--force', is_flag=True, default=False,
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-d', '--debug', is_flag=True,
              help='Enable debug mode. Each runnable specifies the debug behavior. \
                    For example for an Experiment, Ray will run in a single thread \
                    allowing user breakpoints')
@click.option('-e', '--env', type=str, default=None,
              help='Verbose console output')
def run(runnable, output, force, debug, env):
    """Execute a runnable config."""

    # Check if previous job exists
    if env:
        env_config = load_config_from_file(env)
        output = env_config.pop('output_path')

    output = os.path.join(os.path.expanduser(output), 'flambe_output')
    if os.path.exists(output):
        if force:
            shutil.rmtree(output)
        else:
            raise ValueError(f"{output} already exists. Use -f, --force to override.")

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
        logging.disable(logging.NOTSET)

    try:
        kwargs = env_config if env else dict()
        kwargs.update({'output_path': output, 'debug': debug})
        runnable_obj = load_config_from_file(runnable)
        runnable_obj.run(Environment(**kwargs))
        print(cl.GR("------------------- Done -------------------"))
    except KeyboardInterrupt:
        print(cl.RE("---- Exiting early (Keyboard Interrupt) ----"))
    except Exception:
        print(traceback.format_exc())
        print(cl.RE("------------------- Error -------------------"))


# ----------------- flambe submit ------------------ #
@click.command()
@click.argument('runnable', type=str, required=True)
@click.argument('name', type=str, required=True)
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
@click.option('-f', '--force', is_flag=True, default=False,
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-d', '--debug', is_flag=True, default=False,
              help='Enable debug mode. Each runnable specifies the debug behavior. \
                    For example for an Experiment, Ray will run in a single thread \
                    allowing user breakpoints')
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Verbose console output')
@click.option('-a', '--attach', is_flag=True, default=False,
              help='Attach after submitting the job.')
def submit(runnable, name, cluster, force, debug, verbose, attach):
    """Submit a job to the cluster, as a YAML config."""
    logging.disable(logging.INFO)
    if is_dev_mode():
        print(cl.RA(ASCII_LOGO_DEV))
        print(cl.BL(f"Location: {get_flambe_repo_location()}\n"))
    else:
        print(cl.RA(ASCII_LOGO))
        print(cl.BL(f"VERSION: {flambe.__version__}\n"))

    cluster = load_config_from_file(cluster)
    cluster.submit(runnable, name, force=force, debug=debug)
    if attach:
        cluster.attach(name)


# ----------------- flambe site ------------------ #
@click.command()
@click.argument('name', type=str, required=False, default='')
@click.option('-c', '--cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT,
              help="Cluster config.")
@click.option('-p', '--port', type=int, default=49558,
              help='Port in which the site will be running url')
def site(name, cluster, port):
    """Launch a Web UI to monitor the activity on the cluster."""
    logging.disable(logging.INFO)
    cluster = load_config_from_file(cluster)
    try:
        cluster.launch_site(port=port, name=name)
    except KeyboardInterrupt:
        logging.disable(logging.ERROR)


if __name__ == '__main__':
    cli.add_command(up)
    cli.add_command(down)
    cli.add_command(kill)
    cli.add_command(list_cmd, name='ls')
    cli.add_command(exec_cmd, name='exec')
    cli.add_command(attach)
    cli.add_command(run)
    cli.add_command(submit)
    cli.add_command(site)
    cli.add_command(rsync_up, name='rsync-up')
    cli.add_command(rsync_down, name='rsync-down')
    cli(prog_name='flambe')
