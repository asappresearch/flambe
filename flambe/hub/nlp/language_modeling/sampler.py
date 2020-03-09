from typing import Optional, Sequence, Tuple, Iterator, Union
import math
import functools

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from flambe.sampler.sampler import Sampler


class CorpusSampler(Sampler):
    """Implement a CorpusSampler object.

    This object is useful for iteration over a large corpus of
    text in an ordered way. It takes as input a dataset with
    a single example containing the sequence of tokens and will yield
    batches that contain both source sequences of tensors corresponding
    to the Corpus's text, and these same sequences shifted by one as
    the target.

    """

    def __init__(self,
                 batch_size: int = 128,
                 unroll_size: int = 128,
                 n_workers: int = 0,
                 pin_memory: bool = False,
                 downsample: Optional[float] = None,
                 drop_last: bool = True) -> None:
        """Initialize the CorpusSampler object.

        Parameters
        ----------
        batch_size : int, optional
            The batch size to use. Default ``128``.
        unroll_size: int, optional
            Make every sequence this length. Default ``128``.
        n_workers : int, optional
            Number of workers to pass to the DataLoader
            (the default is 0, which means the main process)
        pin_memory : bool, optional
            Pin the memory when using cuda (the default is False)
        downsample: float, optional
            Percentage of the data to downsample to
        drop_last: bool, optional
            Set to True to drop the last incomplete batch if the dataset
            size is not divisible by the batch size.
            (the default is False)

        """
        self.unroll_size = unroll_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.downsample = downsample

    @staticmethod
    def collate_fn(data: Sequence[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Create a batch from data.

        Parameters
        ----------
        data : Sequence[Tuple[Tensor, Tensor]]
            List of (source, target) tuples.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Source and target Tensors.

        """
        x, y = zip(*data)
        return torch.stack(x).t(), torch.stack(y).t()

    def sample(self,
               data: Sequence[Sequence[torch.Tensor]],
               start_iter: int = 0) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Sample from the list of features and yields batches.

        Parameters
        ----------
        data: Sequence[Sequence[Tensor, ...]]
            The input data to sample from
        n_epochs: int, optional
            The number of epochs to run in the output iterator.
            Use -1 to run infinitely.

        Yields
        ------
        Iterator[Tuple[Tensor]]
            A batch of data, as a tuple of Tensors

        """
        if len(data) == 0:
            raise ValueError("No examples provided")
        elif len(data) > 1:
            raise ValueError("Expected a single input example")

        tensor = data[0][0]  # First example, first column

        if self.downsample:
            if not (0 < self.downsample <= 1):
                raise ValueError("Downsample value should be in the range (0, 1]")
            tensor = tensor[:int(self.downsample * tensor.size(0))]

        # Organize batch-wise
        final_length = (tensor.size(0) - 1) // self.batch_size * self.batch_size
        x = torch.reshape(tensor[:final_length], (self.batch_size, -1)).t()
        y = torch.reshape(tensor[1:final_length + 1], (self.batch_size, -1)).t()

        loader = DataLoader(dataset=torch.utils.data.TensorDataset(x, y),
                            collate_fn=self.collate_fn,
                            shuffle=False,
                            batch_size=self.unroll_size,
                            num_workers=self.n_workers,
                            pin_memory=self.pin_memory,
                            drop_last=self.drop_last)
        it = iter(loader)
        for i in range(0, start_iter):
            next(it)
        yield from it

    def length(self, data: Sequence[Sequence[torch.Tensor]]) -> int:
        """Return the number of batches in the sampler.

        Parameters
        ----------
        data: Sequence[Sequence[torch.Tensor, ...]]
            The input data to sample from

        Returns
        -------
        int
            The number of batches that would be created per epoch

        """
        tensor = data[0][0]
        if self.drop_last:
            return ((tensor.size(0) - 1) // self.batch_size) // self.unroll_size
        else:
            return math.ceil(((tensor.size(0) - 1) // self.batch_size) / self.unroll_size)


from flambe.sampler.base import collate_fn as base_collate_fn


class QuickThoughtSampler(Sampler):

    def __init__(self,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 pad_index: Union[int, Sequence[int]] = 0,
                 n_workers: int = 0,
                 pin_memory: bool = False,
                 drop_last: bool = True) -> None:
        self.pad = pad_index
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_workers = n_workers
        self.pin_memory = pin_memory

    @staticmethod
    def collate_fn(data: Sequence[Tuple[Tensor, Tensor]], pad) -> Tuple[Tensor, Tensor]:
        data_and_idx = zip(data, torch.arange(len(data)).long())
        # TODO if need? in tao's implementation left and right are clipped at 128, 64 respectively
        d = [(torch.tensor(row[0]), torch.tensor(row[1]), label) for row, label in data_and_idx]
        return base_collate_fn(d, pad)

    def sample(self,
               data: Sequence[Sequence[torch.Tensor]],
               start_iter: int = 0) -> Iterator[Tuple[torch.Tensor, ...]]:
        if len(data) == 0:
            raise ValueError("No examples provided")

        def examples_generator():
            for convo in data:
                utts = convo[0]  # utterance column
                for l, r in zip(range(0, len(utts)-1), range(1, len(utts))):
                    yield utts[l], utts[r]
        d = list(examples_generator())
        # import pdb; pdb.set_trace()
        collate_fn_p = functools.partial(self.collate_fn, pad=self.pad)
        loader = DataLoader(dataset=d,
                            collate_fn=collate_fn_p,
                            shuffle=self.shuffle,
                            batch_size=self.batch_size,
                            num_workers=self.n_workers,
                            pin_memory=self.pin_memory,
                            drop_last=self.drop_last)
        it = iter(loader)
        for i in range(0, start_iter):
            next(it)
        yield from it

    def length(self, data: Sequence[Sequence[torch.Tensor]]) -> int:
        def examples_generator():
            for convo in data:
                utts = convo[0]  # utterance column
                for l, r in zip(range(0, len(utts)-1), range(1, len(utts))):
                    yield utts[l], utts[r]
        count = 0
        for _ in examples_generator(): count += 1
        return count
