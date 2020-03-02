from typing import List

from flambe.tokenizer import Tokenizer
import fastBPE


class BPETokenizer(Tokenizer):
    """Implement a subword level tokenizer using
       byte pair encoding.  Tokenization is done using
       fastBPE (https://github.com/glample/fastBPE) and
       requires a fastBPE codes file.

    """

    def __init__(self, codes_path: str) -> None:
        """Initialize the tokenizer.

        Parameters
        ----------
        codes_path : str
            Path to codes file created using
            fastBPE.

        """
        self.bpe = fastBPE.fastBPE(codes_path)

    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example.

        Parameters
        ----------
        example : str
            The input example, as a string

        Returns
        -------
        List[str]
            The output subword tokens, as a list of strings

        """
        return self.bpe.apply([example])[0].split(" ")
