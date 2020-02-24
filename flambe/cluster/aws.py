from typing import List, Dict, Optional, Any
from ruamel.yaml import YAML
import tempfile
import copy

from ray.autoscaler.commands import teardown_cluster

from flambe.const import AWS_AMI
from flambe.cluster.cluster import Cluster


class AWSCluster(Cluster):
    """An AWS EC2 cluster."""

    def __init__(self,
                 name: str,
                 head_node_type: str,
                 worker_node_type: str,
                 region: str,
                 security_group: str,
                 subnet_id: str,
                 key_name: str,
                 iam_profile: str,
                 ssh_user: str = 'ubuntu',
                 availability_zone: Optional[List[str]] = None,
                 tags: Dict[str, str] = None,
                 use_spot: bool = False,
                 spot_max_price: Optional[float] = None,
                 head_node_volume_type: str = 'gp2',
                 worker_node_volume_type: str = 'gp2',
                 head_node_volume_size: int = 100,
                 worker_node_volume_size: int = 100,
                 head_node_ami: str = None,
                 worker_node_ami: str = None,
                 dedicated: bool = False,
                 terminate_on_down: bool = False, **kwargs) -> None:
        """Initialize an AWS cluster.

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
        security_group: str
            The security group to use to create the instances.
        subnet_id: str
            The subnet ID to use.
        key_name: str
            The key name that will be used to connect into the instance.
        iam_profile: str
            The key name that will be used to connect into the instance.
        availability_zone: List[str], optional
            An optional list of availability_zones to use.
        tags: Dict[str, str], optional
            A dictionary of tags that will be added to all nodes.
        use_spot: bool, optional
            Whether to use spot instances for worker nodes.
            Default ``False``.
        spot_max_price: float, optional
            The maximum price per hour to spend on a spot instance.
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
        dedicated: bool, optional
            Wether all instances should be dedicated or shared.
        terminate_on_down: bool, optional
            Wether instances should be terminated on flambe down.
            Default ``False``.

        See flambe.cluster.Cluster for extra arguments.

        """
        head_node_ami = head_node_ami or AWS_AMI
        worker_node_ami = worker_node_ami or AWS_AMI

        config: Dict[str, Any] = {
            'provider': {
                'type': 'aws',
                'region': region,
                'cache_stopped_nodes': terminate_on_down
            },
            'head_node': {
                'KeyName': key_name,
                'InstanceType': head_node_type,
                'ImageId': head_node_ami,
                'SubnetIds': [subnet_id],
                'SecurityGroupIds': [security_group],
                'IamInstanceProfile': {
                    'Name': iam_profile
                },
                'BlockDeviceMappings': [
                    {
                        'DeviceName': '/dev/sda1',
                        'Ebs': {
                            'VolumeType': head_node_volume_type,
                            'VolumeSize': head_node_volume_size
                        }
                    }
                ]
            },
            'worker_nodes': {
                'KeyName': key_name,
                'InstanceType': worker_node_type,
                'ImageId': worker_node_ami,
                'IamInstanceProfile': {
                    'Name': iam_profile
                },
                'SubnetIds': [subnet_id],
                'SecurityGroupIds': [security_group],
                'BlockDeviceMappings': [
                    {
                        'DeviceName': '/dev/sda1',
                        'Ebs': {
                            'VolumeType': worker_node_volume_type,
                            'VolumeSize': worker_node_volume_size
                        }
                    }
                ]
            }
        }
        if tags is not None:
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            tag_specification = [
                {
                    "ResourceType": "instance",
                    "Tags": tag_list
                },
                {
                    "ResourceType": "volume",
                    "Tags": tag_list
                }
            ]
            config['head_node']['TagSpecifications'] = tag_specification  # type: ignore
            config['worker_nodes']['TagSpecifications'] = tag_specification  # type: ignore

        if availability_zone:
            config['provider']['availability_zone'] = ','.join(availability_zone)  # type: ignore

        if use_spot:
            spot = {
                'InstanceMarketOptions': {
                    'MarketType': 'spot'
                }
            }
            if spot_max_price:
                spot_options = {
                    'MaxPrice': spot_max_price
                }
                spot['InstanceMarketOptions']['SpotOptions'] = spot_options  # type: ignore

            config['worker_nodes']['InstanceMarketOptions'] = spot  # type: ignore

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

        super().__init__(name=name, ssh_user=ssh_user, extra=config, **kwargs)

    def down(self, yes: bool = False, workers_only: bool = False, destroy: bool = False):
        """Teardown the cluster.

        Parameters
        ----------
        yes : bool, optional
            Tear the cluster down.
        workers_only : bool, optional
            Kill only worker nodes, by default False.
        destroy: boo, optional
            Whether to terminate the instances

        """
        config: Dict[str, Any] = copy.deepcopy(self.config)
        config['provider']['cache_stopped_nodes'] = not destroy
        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(config, fp)
            teardown_cluster(fp.name, yes, workers_only, None)
