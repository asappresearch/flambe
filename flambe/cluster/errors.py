
class ClusterError(Exception):
    """Error raised in case of any unexpected error in the Ray cluster.

    """
    pass


class ClusterConfigurationError(Exception):
    """Error raised when the configuration of the Cluster is not valid.

    """
    pass
