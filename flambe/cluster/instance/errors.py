
class RemoteCommandError(Exception):
    """Error raised when any remote command/script fail in an Instance.

    """
    pass


class SSHConnectingError(Exception):
    """Error raised when opening a SSH connection fails.

    """
    pass


class MissingAuthError(Exception):
    """Error raised when there is missing authentication information.

    """
    pass


class RemoteFileTransferError(Exception):
    """Error raised when sending a local file to an Instance fails.

    """
    pass
