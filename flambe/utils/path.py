import os
from typing import Dict


def rel_to_abs_paths(d: Dict[str, str]) -> Dict[str, str]:
    """Convert relative paths to absolute paths.

    Parameters
    ----------
    d: Dict[str, str]
        A dict from name -> path.

    Returns
    -------
    Dict[str, str]
        The same dict received as parameter with relative paths
        replaced with absolute.

    """
    ret = d.copy()
    for k, v in ret.items():
        if os.path.exists(v) and not os.path.isabs(v):
            ret[k] = os.path.abspath(v)
    return ret
