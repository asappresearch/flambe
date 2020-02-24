from typing import List, Optional

from flambe.const import GCP_AMI
from flambe.cluster.cluster import Cluster


class GCPCluster(Cluster):
    """A GCP cluster."""

    def __init__(self,
                 name: str,
                 head_node_type: str,
                 worker_node_type: str,
                 region: str,
                 subnet: str,
                 availability_zone: Optional[List[str]] = None,
                 preemptible: bool = False,
                 head_node_volume_type: str = 'gp2',
                 worker_node_volume_type: str = 'gp2',
                 head_node_volume_size: int = 100,
                 worker_node_volume_size: int = 100,
                 head_node_ami: str = None,
                 worker_node_ami: str = None, **kwargs) -> None:
        """Initialize a Google Cloud Computer cluster.

        Parameters
        ----------
        name: str
            The unique name for the cluster
        head_node_type : str
            The type of instance to use for the Orchestrator Instances.
            This may not be a GPU instances. At least a "t2.small"
            instance is recommended.
        worker_node_type : str
            The type of instance to use for the Factory Instances.
            GPU instances are required for AWS the AWSCluster.
            "p2" and "p3" instances are recommended.
        region: str
            The region name to use.
        subnet: str
            The subnet ID to use.
        availability_zone: List[str], optional
            An optional list of availability_zones to use.
        preemptible: bool, optional
            Whether to use spot instances for worker nodes.
            Default ``False``.
        head_node_volume_type: str, optional
            The type of volume in AWS to use. Only 'gp2' and 'io1' are
            currently available. If 'io1' is used, then IOPS will be
            fixed to 5000. IMPORTANT: 'io1' volumes are significantly
            more expensive than 'gp2' volumes.
            Defaults to 'gp2'.
        worker_node_volume_type: str, optional
            The type of volume in AWS to use. Only 'gp2' and 'io1' are
            currently available. If 'io1' is used, then IOPS will be
            fixed to 5000. IMPORTANT: 'io1' volumes are significantly
            more expensive than 'gp2' volumes.
            Defaults to 'gp2'.
        head_node_volume_size: int, optional
            The disk size in GB that all hosts will contain.
            Defaults to 100 GB.
        worker_node_volume_size: int, optional
            The disk size in GB that all hosts will contain.
            Defaults to 100 GB.
        head_node_ami: str, optional
            The AMI to be used for the head nodes.
        worker_node_ami: str, optional
            The AMI to be used for the worker nodes.

        See flambe.cluster.Cluster for extra arguments.

        """
        head_node_ami = head_node_ami or GCP_AMI
        worker_node_ami = worker_node_ami or GCP_AMI

        config = {
            'provider': {
                'type': 'gcp',
                'region': region,
                'project_id': None  # Globally unique project id
            },
            'head_node': {
                'machineType': head_node_type,
                'disks': [
                    {
                        'boot': True,
                        'autoDelete': True,
                        'type': 'PERSISTENT',
                        'initializeParams': {
                            'diskSizeGb': head_node_volume_size,
                            'sourceImage': head_node_ami
                        }
                    }
                ],
                'networkInterfaces': [
                    {
                        'kind': 'compute',
                        'subnetwork': subnet,
                        'aliasIpRanges': []
                    }
                ]
            },
            'worker_node': {
                'machineType': worker_node_type,
                'disks': [
                    {
                        'boot': True,
                        'autoDelete': True,
                        'type': 'PERSISTENT',
                        'initializeParams': {
                            'diskSizeGb': worker_node_volume_size,
                            'sourceImage': worker_node_ami
                        }
                    }
                ],
                'networkInterfaces': [
                    {
                        'kind': 'compute',
                        'subnetwork': subnet,
                        'aliasIpRanges': []
                    }
                ]
            }

        }

        if preemptible:
            config['worker_node']['scheduling'] = [{  # type: ignore
                'preemptible': True
            }]

        if availability_zone:
            config['provider']['availability_zone'] = availability_zone  # type: ignore

        # Command to start ray on the head and worker nodes
        config['head_start_ray_commands'] = [
            'ray stop',
            'ulimit -n 65536; ray start --head --redis-port=6379 --include-webui 1 \
                --object-manager-port=8076 --autoscaling-config=~/ray_bootstrap_config.yaml'
        ]
        config['worker_start_ray_commands'] = [
            'ray stop',
            'ulimit -n 65536; ray start --address=$RAY_HEAD_IP:6379 --object-manager-port=8076'
        ]

        super().__init__(name, extra=config, **kwargs)
