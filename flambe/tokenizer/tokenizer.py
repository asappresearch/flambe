from abc import abstractmethod
from typing import List

from flambe import Component


class Tokenizer(Component):
    """Base interface to a Tokenizer object.

    Tokenizers implement the `tokenize` method, which takes a
    string as input and produces a list of strings as output.

    """

    @abstractmethod
    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example.

        Parameters
        ----------
        example : str
            The input example, as a string

        Returns
        -------
        List[str]
            The output tokens, as a list of strings

        """
        pass

    def __call__(self, example: str):
        """Make a tokenizer callable."""
        return self.tokenize(example)
