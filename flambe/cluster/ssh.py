"""Implementation of the Manager for SSH hosts"""

import logging

from typing import List, TypeVar, Union, Optional

from flambe.cluster import instance
from flambe.cluster.cluster import Cluster, FactoryInsT

import os


logger = logging.getLogger(__name__)

FactoryT = TypeVar("FactoryT", instance.CPUFactoryInstance, instance.GPUFactoryInstance)


class SSHCluster(Cluster):
    """The SSH Manager needs to be used when having running instances.

    For example when having on-prem hardware or just a couple
    of AWS EC2 instances running.

    When using this cluster, the user needs to specify the IPs of
    the machines to use, both the public one and private one.

    """

    def __init__(self,
                 name: str,
                 orchestrator_ip: Union[str, List[str]],
                 factories_ips: Union[List[str], List[List[str]]],
                 key: str,
                 username: str,
                 remote_context=None,
                 use_public: bool = True,
                 setup_cmds: Optional[List[str]] = None) -> None:
        """Initialize the SSHCluster."""
        super().__init__(name, len(factories_ips), key, username, setup_cmds)
        self.orchestrator_ip = orchestrator_ip
        self.factories_ips = factories_ips
        self.remote_context = remote_context

        self.use_public = use_public

        if remote_context:
            self.cluster_id = self.remote_context.cluster_id

    def load_all_instances(self, exp_name: str = None, force: bool = False) -> None:
        """This manager assumed that instances are running.

        This method loads the Python objects to the manager's variables.

        Parameters
        ----------
        exp_name: str
            The name of the experiment
        force: bool
            Whether to override the current experiment of the same name

        """
        if isinstance(self.orchestrator_ip, list):
            self.orchestrator = self.get_orchestrator(self.orchestrator_ip[0],
                                                      self.orchestrator_ip[1],
                                                      use_public=self.use_public)
        else:
            self.orchestrator = self.get_orchestrator(self.orchestrator_ip,
                                                      use_public=self.use_public)

        aux: FactoryInsT
        for each in self.factories_ips:
            if isinstance(each, list):
                factory = self.get_factory(each[0], each[1], use_public=self.use_public)
                if factory.contains_gpu():
                    factory = self.get_gpu_factory(each[0], each[1], use_public=self.use_public)
            else:
                factory = self.get_factory(each, use_public=self.use_public)
                if factory.contains_gpu():
                    factory = self.get_gpu_factory(each, use_public=self.use_public)

            self.factories.append(factory)

    def rollback_env(self) -> None:
        pass

    def rsync_hosts(self):
        """Rsyncs the host's result folders.

        First, it rsyncs all worker folders to the orchestrator main
        folder. After that, so that every worker gets the last changes,
        the orchestrator rsync with all of them.

        """
        if not self.remote_context:
            logger.error("Can't rsyn without a remote context")
            return

        exclude = ["state.pkl"]

        orch = self.orchestrator
        orch_save_path = os.path.join(f"{orch.get_home_path()}", self.remote_context.save_folder)
        orch_loc = f"{orch_save_path}"

        for f in self.factories:
            f_save_path = os.path.join(f"{orch.get_home_path()}", self.remote_context.save_folder)
            f_loc = f"{f.username}@{f.private_host}:{f_save_path}"
            orch.rsync_folder(f_loc, orch_loc, exclude)

        for f in self.factories:
            f_save_path = os.path.join(f"{f.get_home_path()}", self.remote_context.save_folder)
            f_loc = f"{f.username}@{f.private_host}:{f_save_path}"
            orch.rsync_folder(orch_loc, f_loc, exclude)
