from flambe.cluster.cluster import Cluster


class KubernetesCluster(Cluster):
    """A Kubernetes cluster."""

    def __init__(self,
                 name: str,
                 head_node_num_cpu: str,
                 worker_node_num_cpu: str,
                 key_name: str,
                 head_node_cpu: str = 'gp2',
                 worker_node_volume_type: str = 'gp2',
                 head_node_volume_size: int = 100,
                 worker_node_volume_size: int = 100,
                 head_node_ami: str = None,
                 worker_node_ami: str = None, **kwargs) -> None:
        """Initialize a Kubernetes cluster.

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
        key_name: str
            The key name that will be used to connect into the instance.
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
        super().__init__(name, **kwargs)

        config = {
            'provider': {
                'type': 'kubernetes',
                'use_internal_ips': True,
                'namespace': 'flambe',
                'autoscaler_service_account': {
                    'apiVersion': 'v1',
                    'kind': 'ServiceAccount',
                    'metadata': {
                        'name': 'autoscaler'
                    }
                },
                'autoscaler_role': {
                    'kind': 'Role',
                    'apiVersion': 'rbac.authorization.k8s.io/v1',
                    'metadata': {
                        'name': 'autoscaler'
                    },
                    'rules': [{
                        'apiGroups': [""],
                        'resources': ["pods", "pods/status", "pods/exec"],
                        'verbs': ["get", "watch", "list", "create", "delete", "patch"]
                    }]
                },
                'autoscaler_role_binding': {
                    'apiVersion': 'rbac.authorization.k8s.io/v1',
                    'kind': 'RoleBinding',
                    'metadata': {
                        'name': 'autoscaler'
                    },
                    'subjects': [{
                        'kind': 'ServiceAccount',
                        'name': 'autoscaler',
                        'roleRef': {
                            'kind': 'Role',
                            'name': 'autoscaler',
                            'apiGroup': 'rbac.authorization.k8s.io'
                        }
                    }]
                }
            },
            # Kubernetes pod config for the head node pod.
            'head_node': {
                'apiVersion': 'v1',
                'kind': 'Pod',
                'metadata': {
                    'generateName': 'flambe-head-'
                },
                'spec': {
                    'serviceAccountName': 'autoscaler',
                    'restartPolicy': 'Never',  # Restart is not supported
                    # This volume allocates shared memory for Ray to use for its plasma
                    # object store. If you do not provide this, Ray will fall back to
                    # /tmp which cause slowdowns if is not a shared memory volume.
                    'volumes': [{
                        'name': 'dshm',
                        'emptyDir': {
                            'medium': 'Memory'
                        }
                    }],
                    'containers': [{
                        'name': 'flambe-node',
                        'imagePullPolicy': 'Always',
                        # You are free (and encouraged) to use your own container image,
                        # but it should have the following installed:
                        #   - rsync (used for `ray rsync` commands and file mounts)
                        #   - screen (used for `ray attach`)
                        #   - kubectl (used by the autoscaler to manage worker pods)
                        'image': 'rayproject/autoscaler',
                        # explicitly killed.
                        'command': ["/bin/bash", "-c", "--"],  # Do not change
                        'args': ["trap : TERM INT; sleep infinity & wait;"],
                        'ports': [
                            {'containerPort': 6379},
                            {'containerPort': 6380},
                            {'containerPort': 6381},
                            {'containerPort': 12345},
                            {'containerPort': 12346}
                        ]
                    }],
                    'volumeMounts': [{
                        'mountPath': '/dev/shm',
                        'name': 'dshm'
                    }],
                    'resources': {
                        'requests': {
                            'cpu': 'head_node_cpu',
                            'memory': 'head_node_memory'
                        },
                        'limits': {
                            'memory': 'head_node_memory'
                        }
                    },
                    'env': [{
                        'name': 'MY_CPU_REQUEST',
                        'valueFrom': {
                            'resourceFieldRef': {
                                'resource': 'requests.cpu'
                            }
                        }
                    }]
                }
            },
            'worker_nodes': {
                'apiVersion': 'v1',
                'kind': 'Pod',
                'metadata': {
                    'generateName': 'flambe-worker-'
                },
                'spec': {
                    'serviceAccountName': 'default',
                    'restartPolicy': 'Never',  # Restart is not supported
                    'volumes': [{
                        'name': 'dshm',
                        'emptyDir': {
                            'medium': 'Memory'
                        }
                    }],
                    'containers': [{
                        'name': 'flambe-node',
                        'imagePullPolicy': 'Always',
                        # You are free (and encouraged) to use your own container image,
                        # but it should have the following installed:
                        #   - rsync (used for `ray rsync` commands and file mounts)
                        #   - screen (used for `ray attach`)
                        #   - kubectl (used by the autoscaler to manage worker pods)
                        'image': 'rayproject/autoscaler',
                        'command': ["/bin/bash", "-c", "--"],  # Do not change
                        'args': ["trap : TERM INT; sleep infinity & wait;"],
                        'ports': [
                            {'containerPort': 12345},
                            {'containerPort': 12346}
                        ]
                    }],
                    'volumeMounts': [{
                        'mountPath': '/dev/shm',
                        'name': 'dshm'
                    }],
                    'resources': {
                        'requests': {
                            'cpu': 'head_node_cpu',
                            'memory': 'head_node_memory'
                        },
                        'limits': {
                            'memory': 'head_node_memory'
                        }
                    },
                    'env': [{
                        'name': 'MY_CPU_REQUEST',
                        'valueFrom': {
                            'resourceFieldRef': {
                                'resource': 'requests.cpu'
                            }
                        }
                    }]
                }
            }
        }

        # Command to start ray on the head and worker nodes
        config['head_start_ray_commands'] = [
            'ray stop',
            'ulimit -n 65536; ray start --head --num-cpus=$MY_CPU_REQUEST --redis-port=6379 \
                --object-manager-port=8076 --autoscaling-config=~/ray_bootstrap_config.yaml'
        ]
        config['worker_start_ray_commands'] = [
            'ray stop',
            'ulimit -n 65536; ray start --num-cpus=$MY_CPU_REQUEST \
                --address=$RAY_HEAD_IP:6379 --object-manager-port=8076'
        ]

        self.config.update(config)
