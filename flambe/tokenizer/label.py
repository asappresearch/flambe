
from typing import Optional, List

from flambe.tokenizer import Tokenizer


class LabelTokenizer(Tokenizer):
    """Base label tokenizer.

    This object tokenizes string labels into a list of a single or
    multiple elements, depending on the provided separator.

    """
    def __init__(self, multilabel_sep: Optional[str] = None) -> None:
        """Initialize the tokenizer.

        Parameters
        ----------
        multilabel_sep : Optional[str], optional
            Used to split multi label inputs, if given

        """
        self.multilabel_sep = multilabel_sep

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
        sep = self.multilabel_sep
        return example.split(sep) if sep else [example]
