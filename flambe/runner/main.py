import logging
import click

import torch

import flambe
from flambe.const import FLAMBE_CLUSTER_DEFAULT
from flambe.logging import setup_global_logging
from flambe.logging import coloredlogs as cl
from flambe.runner.utils import is_dev_mode, get_flambe_repo_location
from flambe.logo import ASCII_LOGO, ASCII_LOGO_DEV
from flambe.compile.yaml import load_config_from_file as load_config


logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


# ----------------- flambe up ------------------ #
@click.command()
@click.argument('cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT)
def up(cluster):
    """Launch / update the cluster based on the given config"""
    cluster = load_config(cluster)
    cluster.up()


# ----------------- flambe down ------------------ #
@click.command()
@click.argument('cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT)
def down(cluster):
    """Launch / update the cluster based on the given config"""
    cluster = load_config(cluster)
    cluster.down()


# ----------------- flambe exec ------------------ #
@click.command()
@click.argument('cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT)
@click.argument('command', type=str)
def exec(cluster, command):
    """Execute a command on the head node."""
    cluster = load_config(cluster)
    cluster.exec(command=command)


# ----------------- flambe list ------------------ #
@click.command()
@click.argument('cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT)
def list(cluster):
    """Attach to a tmux session on the cluster."""
    cluster = load_config(cluster)
    cluster.list()


# ----------------- flambe attach ------------------ #
@click.command()
@click.argument('cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT)
def attach(cluster):
    """Attach to a tmux session on the cluster."""
    cluster = load_config(cluster)
    cluster.attach()


# ----------------- flambe kill ------------------ #
@click.command()
@click.argument('cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT)
def kill(cluster, name):
    """Attach to a tmux session on the cluster."""
    cluster = load_config(cluster)
    cluster.kill(name)


# ----------------- flambe run ------------------ #
@click.command()
@click.argument('runnable', type=str, required=True)
@click.option('-o', '--output', default='flambe__output',
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-f', '--force', is_flag=True, default=False,
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-d', '--debug', is_flag=True,
              help='Enable debug mode. Each runnable specifies the debug behavior. \
                    For example for an Experiment, Ray will run in a single thread \
                    allowing user breakpoints')
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Verbose console output')
def run(runnable, output, force, debug, verbose):
    """Execute command based on given config"""
    # torch.multiprocessing exists, ignore mypy
    # TODO: investigate if this is actually needed
    torch.multiprocessing.set_start_method('fork', force=True)  # type: ignore

    # Setup logging
    setup_global_logging(logging.INFO if not verbose else logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Check if dev mode
    if is_dev_mode():
        print(cl.RA(ASCII_LOGO_DEV))
        print(cl.BL(f"Location: {get_flambe_repo_location()}\n"))
    else:
        print(cl.RA(ASCII_LOGO))
        print(cl.BL(f"VERSION: {flambe.__version__}\n"))

    if args.debug:
        print(cl.YE(f"Debug mode activated\n"))

    try:
        runnable = load_config(config)
        runnable.run(Environment(output, force=force, debug=debug))
        logger.info(cl.GR("------------------- Done -------------------"))
    except KeyboardInterrupt:
        logger.info(cl.RE("---- Exiting early (Keyboard Interrupt) ----"))


# ----------------- flambe submit ------------------ #
@click.command()
@click.argument('runnable', type=str, required=True)
@click.argument('cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT)
@click.option('-f', '--force', is_flag=True, default=False,
              help='Override existing job with this name. Be careful \
                    when using this flag as it could have undesired effects.')
@click.option('-d', '--debug', is_flag=True, default=False,
              help='Enable debug mode. Each runnable specifies the debug behavior. \
                    For example for an Experiment, Ray will run in a single thread \
                    allowing user breakpoints')
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Verbose console output')
def submit(cluster, runnnable, name, force, debug, verbos, args):
    """Submit a job to the cluster."""
    if is_dev_mode():
        print(cl.RA(ASCII_LOGO_DEV))
        print(cl.BL(f"Location: {get_flambe_repo_location()}\n"))
    else:
        print(cl.RA(ASCII_LOGO))
        print(cl.BL(f"VERSION: {flambe.__version__}\n"))

    cluster = load_config(cluster)
    cluster.submit(runnable, name, force=force, debug=debug)


# ----------------- flambe site ------------------ #
@click.command()
@click.argument('cluster', type=str, default=FLAMBE_CLUSTER_DEFAULT)
@click.option('--name', type=str, default='',
              help='The name of the job to inspect')
@click.option('--port', type=int, default=49558,
              help='Port in which the site will be running url')
def site(config, name, port, launch_tensorboard):
    cluster = yaml.load(config)
    cluster.launch_site(port=port, name=name)


if __name__ == '__main__':
    cli.add_command(up)
    cli.add_command(down)
    cli.add_command(submit)
    cli.add_command(site)
    cli()
