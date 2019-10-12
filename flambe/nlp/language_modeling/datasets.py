from typing import List, Tuple, Optional, Union, Dict
from zipfile import ZipFile
from io import BytesIO
import requests

from flambe.dataset import TabularDataset
from flambe.field import Field


class PTBDataset(TabularDataset):
    """The official PTB dataset."""

    PTB_URL = "https://raw.githubusercontent.com/yoonkim/lstm-char-cnn/master/data/ptb/"

    def __init__(self,
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the PTBDataset builtin."""
        train_path = self.PTB_URL + "train.txt"
        val_path = self.PTB_URL + "valid.txt"
        test_path = self.PTB_URL + "test.txt"

        train, _ = self._load_file(train_path)
        val, _ = self._load_file(val_path)
        test, _ = self._load_file(test_path)

        super().__init__(train, val, test, cache=cache, transform=transform)

    @classmethod
    def _load_file(cls,
                   path: str,
                   sep: Optional[str] = '\t',
                   header: Optional[str] = None,
                   columns: Optional[Union[List[str], List[int]]] = None,
                   encoding: Optional[str] = 'utf-8') -> Tuple[List[Tuple], Optional[List[str]]]:
        """Load data from the given path."""
        data, named_cols = super()._load_file(path, sep, header, columns)
        return [(d[0][:],) for d in data], named_cols


class Wiki103(TabularDataset):
    """The official WikiText103 dataset."""

    WIKI_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"

    def __init__(self,
                 word_wrap: int = 128,
                 end_of_line_token: str = '</s>',
                 cache: bool = False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the Wiki103 builtin.

        Parameters
        ----------
        word_wrap: int, Optional
            Make every sequence of this length. Default ``128``.
        end_of_line_token: str, Optional
            Token added at the end of every line.

        see TabularDataset for other arguments.

        """
        self.word_wrap = word_wrap
        self.eol = end_of_line_token
        response = requests.get(self.WIKI_URL, stream=True)
        with ZipFile(BytesIO(response.content), 'r') as z:
            train = self._process(z.read('wikitext-103/wiki.train.tokens'))
            val = self._process(z.read('wikitext-103/wiki.valid.tokens'))
            test = self._process(z.read('wikitext-103/wiki.test.tokens'))

        super().__init__(train, val, test, cache=cache, transform=transform)

    def _process(self, file) -> List[str]:
        split = file.decode('utf-8').replace('\\n', self.eol).split()
        steps = range(0, len(split) - self.word_wrap, self.word_wrap)
        text = [" ".join(split[i:i + self.word_wrap]) for i in steps]
        return text
