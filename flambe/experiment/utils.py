import copy
import os
from typing import Dict, Mapping, Any, Optional, Set

import torch
from ruamel.yaml.compat import StringIO
from ruamel import yaml as original_yaml

from flambe.compile import Link
from flambe.compile import Schema as Schema
from flambe.runnable.error import LinkError


def check_links(blocks: Dict[str, Schema],
                global_vars: Optional[Dict[str, Any]] = None) -> None:
    """Check validity of links between blocks.

    Ensures dependency order, and that only Comparable
    blocks are being reduced through a LinkBest object.

    Parameters
    ----------
    blocks : OrderedDict[str, Schema[Component]]
        The blocks to check, in order

    Raises
    ------
    LinkError
        On undeclared blocks (i.e not the right config order)
    ProtocolError
        Attempt to reduce a non-comparable block

    """
    visited: Set[str] = set()
    if global_vars is not None:
        visited |= global_vars.keys()

    def helper(block):
        """Explore block"""
        for _, value in block.items():
            # Check link order
            if isinstance(value, Link):
                target_block_id = value.root_schema
                if target_block_id not in visited:
                    raise LinkError(block_id, target_block_id)

            # Check recurse
            if isinstance(value, Mapping):
                helper(value)

    for block_id, block in blocks.items():
        visited.add(block_id)
        helper(block)


def get_non_remote_config(experiment):
    """Returns a copy of the original config file without
    the remote configuration

    Parameters
    ----------
    experiment : Experiment
        The experiment object

    """
    new_experiment = copy.deepcopy(experiment)
    # Remove manager
    experiment.manager = None

    with StringIO() as s:
        native_yaml = original_yaml.YAML()
        native_yaml.dump(new_experiment, s)
        return s.getvalue()


def local_has_gpu() -> bool:
    """Returns is local process has GPU

    Returns
    -------
    bool

    """
    return torch.cuda.is_available()


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
