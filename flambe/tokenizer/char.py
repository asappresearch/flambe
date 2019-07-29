from typing import List

from flambe.tokenizer import Tokenizer


class CharTokenizer(Tokenizer):
    """Implement a character level tokenizer."""

    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example.

        Parameters
        ----------
        example : str
            The input example, as a string

        Returns
        -------
        List[str]
            The output charachter tokens, as a list of strings

        """
        return list(example)
