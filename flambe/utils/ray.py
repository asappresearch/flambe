import subprocess
from io import StringIO
import sys
from typing import List

import ray
from ray.autoscaler.updater import SSHCommandRunner

from flambe.runner import Environment


def initialize(env: Environment):
    if not ray.is_initialized():
        if env.remote:
            ray.init("auto", local_mode=env.debug)
        else:
            ray.init(local_mode=env.debug)


def supress_ssh(func):
    """Set SSH level for ray calls to QUIET"""
    def wrapper(self, connect_timeout):
        return func(self, connect_timeout) + ['-o', "LogLevel=QUIET"]
    return wrapper


def supress_rsync(func, supress_all=False):
    """Surpess rsync messages from ray."""
    def wrapper(args, *, stdin=None, stdout=None, stderr=None,
                shell=False, cwd=None, timeout=None):
        if args[0] == 'rsync' or supress_all:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        func(args, stdin=None, stdout=stdout, stderr=stderr,
             shell=False, cwd=None, timeout=None)
    return wrapper


class capture_ray_output(object):
    """Supress the messages coming from the ssh command."""

    def __init__(self, supress_all: bool = False):
        self.out: List = []
        self.supress_all = supress_all

    def __enter__(self):
        self._ssh = SSHCommandRunner.get_default_ssh_options
        self._ssh_all = subprocess.check_call
        self._stdout = sys.stdout

        SSHCommandRunner.get_default_ssh_options = supress_ssh(
            SSHCommandRunner.get_default_ssh_options
        )
        subprocess.check_call = supress_rsync(subprocess.check_call, self.supress_all)
        sys.stdout = self._stringio = StringIO()
        return self.out

    def __exit__(self, *args):
        self.out.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        SSHCommandRunner.get_default_ssh_options = self._ssh
        subprocess.check_call = self._ssh_all
        sys.stdout = self._stdout
