import time
import logging
from typing import Any, Sequence, NamedTuple, Union, Dict, Tuple, Optional

import torch
import numpy

# Officially Supported Logging Data Types


class ScalarT(NamedTuple):
    """A single scalar value

    Supported by TensorboardX

    Parameters
    ----------
    tag: str
        Data identifier
    scalar_value: float
        The scalar value
    global_step: int
        Iteration associated with this value
    walltime: float = time.time()
        Wall clock time associated with this value

    """

    tag: str
    scalar_value: float
    global_step: int
    walltime: float = time.time()

    def __repr__(self) -> str:
        return f'{self.tag}#{self.global_step} = {self.scalar_value} ' \
               + f'({time.strftime("%H:%M:%S", time.localtime(self.walltime))})'


class ScalarsT(NamedTuple):
    """A dictionary mapping tag keys to scalar values

    Supported by TensorboardX

    Parameters
    ----------
    main_tag: str
        Parent name for all the children tags
    tag_scalar_dict: Dict[str, float]
        Mapping from scalar tags to their values
    global_step: int
        Iteration associated with this value
    walltime: float = time.time()
        Wall clock time associated with this value

    """

    main_tag: str
    tag_scalar_dict: Dict[str, float]
    global_step: int
    walltime: float = time.time()

    def __repr__(self) -> str:
        return f'{self.main_tag}#{self.global_step} = {self.tag_scalar_dict} ' \
               + f'({time.strftime("%H:%M:%S", time.localtime(self.walltime))})'


class HistogramT(NamedTuple):
    """A histogram with an array of values

    Supported by TensorboardX

    Parameters
    ----------
    tag: str
        Data identifier
    values: Union[torch.Tensor, numpy.array]
        Values to build histogram
    global_step: int
        Iteration associated with this value
    bins: str
        Determines how bins are made
    walltime: float = time.time()
        Wall clock time associated with this value

    """

    tag: str
    values: Union[torch.Tensor, numpy.array]
    global_step: int
    bins: str  # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
    walltime: float = time.time()

    def __repr__(self) -> str:
        return f'{self.tag}#{self.global_step} = {self.values} ' \
               + f'({time.strftime("%H:%M:%S", time.localtime(self.walltime))})'


class ImageT(NamedTuple):
    """Image message

    Supported by TensorboardX

    Parameters
    ----------
    tag: str
        Data identifier
    img_tensor: Union
        Image tensor to record
    global_step: int
        Iteration associated with this value
    walltime: float
        Wall clock time associated with this value

    """

    tag: str
    img_tensor: Union[torch.Tensor, numpy.array]
    global_step: int
    walltime: float

    def __repr__(self) -> str:
        return f'{self.tag}#{self.global_step} = {self.img_tensor} ' \
               + f'({time.strftime("%H:%M:%S", time.localtime(self.walltime))})'


class TextT(NamedTuple):
    """Text message

    Supported by TensorboardX

    Parameters
    ----------
    tag: str
        Data identifier
    text_string: str
        String to record
    global_step: int
        Iteration associated with this value
    walltime: float
        Wall clock time associated with this value

    """

    tag: str
    text_string: str
    global_step: int
    walltime: float

    def __repr__(self) -> str:
        return f'{self.tag}#{self.global_step} = {self.text_string} ' \
               + f'({time.strftime("%H:%M:%S", time.localtime(self.walltime))})'


class PRCurveT(NamedTuple):
    """PRCurve message

    Supported by TensorboardX

    Parameters
    ----------
    tag: str
        Data identifier
    labels: Union[torch.Tensor, numpy.array]
        Containing 0, 1 values
    predictions: Union[torch.Tensor, numpy.array]
        Containing 0<=x<=1 values. Needs to match labels size
    num_thresholds: int = 127
        The number of thresholds to evaluate. Max value allowed 127.
    weights: Optional[float] = None
        No description provided.
    global_step: int
        Iteration associated with this value
    walltime: float
        Wall clock time associated with this value

    """
    tag: str
    labels: Union[torch.Tensor, numpy.array]
    predictions: Union[torch.Tensor, numpy.array]
    global_step: int
    num_thresholds: int = 127
    weights: Optional[float] = None
    walltime: float = time.time()

    def __repr__(self) -> str:
        return f'{self.tag}#{self.global_step} = ... ' \
               + f'({time.strftime("%H:%M:%S", time.localtime(self.walltime))})'


class EmbeddingT(NamedTuple):
    """Embedding data, including array of vaues and metadata

    Supported by TensorboardX

    Parameters
    ----------
    mat: Union[torch.Tensor, numpy.array]
        A matrix where each row is the feature vector of a data point
    metadata: Sequence[str]
        A list of labels; each element will be converted to string
    label_img: torch.Tensor
        Images corresponding to each data point
    global_step: int
        Iteration associated with this value
    tag: str
        Data identifier
    metadata_header: Sequence[str]

    Shape
    -----
    mat: :math:`(N, D)`
        where N is number of data and D is feature dimension
    label_img: :math:`(N, C, H, W)`

    """

    mat: Union[torch.Tensor, numpy.array]
    metadata: Sequence[str]
    label_img: torch.Tensor
    global_step: int
    tag: str
    metadata_header: Sequence[str]

    def __repr__(self) -> str:
        return f'{self.tag}#{self.global_step} ... ({self.metadata})'


class GraphT(NamedTuple):
    """PyTorch Model with input and other keyword args

    Supported by ModelSave
    NOT YET Supported by TensorboardX

    Attributes
    ----------
    model: torch.nn.Module
        PyTorch Model (should have `forward` and `state_dict` methods)
    input_to_model: torch.autograd.Variable
        Input to the model `forward` call
    verbose: bool = False
        Include extra detail
    kwargs: Dict[str, Any] = {}
        Other kwargs for model recording

    """

    model: torch.nn.Module
    input_to_model: torch.autograd.Variable
    verbose: bool = False
    kwargs: Dict[str, Any] = {}


DATA_TYPES = tuple([ScalarT, ScalarsT, HistogramT, TextT, ImageT, EmbeddingT, GraphT, PRCurveT])


class DataLoggingFilter(logging.Filter):
    """Filters on `DATA_TYPES` otherwise returns `default`

    `filter` returns `self.default` if record is not a `DATA_TYPES`
    type; True if message is a `DATA_TYPES` type not in `dont_include`
    and high enough level; otherwise False

    Parameters
    ----------
    default : bool
        Returned when record is not one `DATA_TYPES`
    level : int
        Minimum level of records that are `DATA_TYPES` to be accepted
    dont_include : Sequence[Type[Any]]
        Types from `DATA_TYPES` to be excluded
    **kwargs : Any
        Additional kwargs to pass to `logging.Filter`

    Attributes
    ----------
    default : bool
        Returned when record is not one `DATA_TYPES`
    level : int
        Minimum level of records that are `DATA_TYPES` to be accepted
    dont_include : Tuple[Type[Any]]
        Types from `DATA_TYPES` to be excluded

    """
    def __init__(self,
                 default: bool = True,
                 level: int = logging.NOTSET,
                 dont_include: Optional[Tuple[type, ...]] = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.default = default
        self.level = level
        self.dont_include = tuple(dont_include) if dont_include is not None else tuple()

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True iff record should be accepted

        Parameters
        ----------
        record : logging.LogRecord
            logging record to be filtered

        Returns
        -------
        bool
            True iff record should be accepted. `self.default` if
            record is not a `DATA_TYPES` type; True if message is a
            `DATA_TYPES` type not in `dont_include` and high enough
            level; otherwise False

        """
        if hasattr(record, 'raw_msg_obj'):
            message = record.raw_msg_obj  # type: ignore
            if isinstance(message, DATA_TYPES):
                if not isinstance(message, self.dont_include) \
                   and record.levelno >= self.level:
                    return True
                return False
            return self.default
        return self.default
