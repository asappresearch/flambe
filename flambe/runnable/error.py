class ProtocolError(Exception):

    def __init__(self, message: str) -> None:
        """Base ProtocolError implementation.

        Parameters
        ----------
        message : str
            The message to display

        """
        self.message = message

    def __repr__(self) -> str:
        """Override output message to show ProtocolError

        Returns
        -------
        str
            The output message

        """
        return f"ProtocolError: {self.message}"


class LinkError(ProtocolError):

    def __init__(self, block_id: str, target_block_id: str) -> None:
        """Link error on undeclared block.

        Parameters
        ----------
        block_id : str
            The block including the link
        target_block_id : str
            The link's target block

        """
        default_message = (
            f"Block '{block_id}' has a link to '{target_block_id}' "
            "which has not yet been declared."
        )
        super().__init__(default_message)


class SearchComponentError(ProtocolError):

    def __init__(self, block_id: str) -> None:
        """Search error on non-computable.

        Parameters
        ----------
        block_id : str
            The block with the wrong type

        """
        default_message = (
            f"Block '{block_id}' is a non-component; "
            "only Component objects can be in the pipeline"
        )
        super().__init__(default_message)


class UnsuccessfulRunnableError(RuntimeError):
    pass


class RunnableFileError(Exception):
    pass


class ResourceError(RunnableFileError):
    pass


class NonExistentResourceError(RunnableFileError):
    pass


class ExistentResourceError(RunnableFileError):
    pass


class ParsingRunnableError(RunnableFileError):
    pass


class TagError(RunnableFileError):
    pass


class MissingSecretsError(Exception):
    pass
