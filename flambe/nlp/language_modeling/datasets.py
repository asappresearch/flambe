from typing import List, Tuple, Optional, Union, Dict

from flambe.dataset import TabularDataset
from flambe.field import Field


class PTBDataset(TabularDataset):
    """The official SST training dataset."""

    PTB_URL = "https://raw.githubusercontent.com/yoonkim/lstm-char-cnn/master/data/ptb/"

    def __init__(self, cache=False,
                 transform: Dict[str, Union[Field, Dict]] = None) -> None:
        """Initialize the SSTDataset builtin."""
        train_path = self.PTB_URL + "train.txt"
        dev_path = self.PTB_URL + "valid.txt"
        test_path = self.PTB_URL + "test.txt"

        train, _ = self._load_file(train_path)
        dev, _ = self._load_file(dev_path)
        test, _ = self._load_file(test_path)

        super().__init__(train, dev, test, cache=cache, transform=transform)

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
