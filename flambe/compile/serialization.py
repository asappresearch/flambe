import dill
from typing import Any

import torch


def save(obj: Any, path: str):
    """Save an object to a file.

    Parameters
    ----------
    obj : Any
        The object to save.
    path : str
        The path to save the object to.

    """
    torch.save(obj, path, pickle_module=dill)


def load(path: str) -> Any:
    """Load from path.

    Parameters
    ----------
    path: str
        The path to load form.

    Returns
    -------
    Any
        The loaded object.

    """
    return torch.load(path, map_location='cpu', pickle_module=dill)
