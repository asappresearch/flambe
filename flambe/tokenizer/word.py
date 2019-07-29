
from typing import Union, List

from nltk import ngrams

from flambe.tokenizer import Tokenizer


class WordTokenizer(Tokenizer):
    """Implement a word level tokenizer."""

    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example.

        Parameters
        ----------
        example : str
            The input example, as a string

        Returns
        -------
        List[str]
            The output word tokens, as a list of strings

        """
        return example.split()


class NGramsTokenizer(Tokenizer):
    """Implement a n-gram tokenizer

    Examples
    --------

    >>> t = NGramsTokenizer(ngrams=2).tokenize("hi how are you?")
    ['hi, how', 'how are', 'are you?']

    >>> t = NGramsTokenizer(ngrams=[1,2]).tokenize("hi how are you?")
    ['hi,', 'how', 'are', 'you?', 'hi, how', 'how are', 'are you?']

    Parameters
    ----------
    ngrams: Union[int, List[int]]
        An int or a list of ints. If it's a list of ints, all n-grams
        (for each int) will be considered in the tokenizer.

    """
    def __init__(self, ngrams: Union[int, List[int]] = 1) -> None:
        """[summary]

        Parameters
        ----------
        ngrams : Union[int, List[int]], optional
            [description], by default 1

        Returns
        -------
        None
            [description]
        """
        self.ngrams = ngrams

    @staticmethod
    def _tokenize(example: str, n: int) -> List[str]:
        """Tokenize an input example using ngrams.

        """
        return list(" ".join(x) if len(x) > 1 else x[0] for x in ngrams(example.split(), n))

    def tokenize(self, example: str) -> List[str]:
        """Tokenize an input example.

        Parameters
        ----------
        example : str
            The input example, as a string.

        Returns
        -------
        List[str]
            The output word tokens, as a list of strings

        """
        if isinstance(self.ngrams, List):
            ret: List[str] = []
            for i in self.ngrams:
                ret.extend(self._tokenize(example, i))
            return ret
        else:
            return NGramsTokenizer._tokenize(example, self.ngrams)
