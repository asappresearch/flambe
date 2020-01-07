import os
import getpass
from flambe.compile.utils import _is_url

import flambe

try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze


DEFAULT_USER_PROVIDER = getpass.getuser
MB = 2**20
WARN_LIMIT_MB = 100


def get_size_MB(path: str) -> float:
    """Return the size of a file/folder in MB.

    Parameters
    ----------
    path: str
        The path to the folder or file

    Returns
    -------
    float
        The size in MB

    """
    accum = 0
    if os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp) and not os.path.islink(fp):
                    accum += os.path.getsize(fp)
    else:
        accum = os.path.getsize(path)
    return accum / MB


def check_system_reqs() -> None:
    """Run system checks and prepare the system before a run.

    This method should:
        * Create folders, files that are needed for flambe
        * Raise errors in case requirements are not met. This should
        run under the SafeExecutionContext, so errors will be handled
        * Warn the user in case something needs attention.

    """
    # Create the flambe folder if it does not exist
    if not os.path.exists(FLAMBE_GLOBAL_FOLDER):
        os.mkdir(FLAMBE_GLOBAL_FOLDER)

    # Check if extensions folder is getting big
    extensions_folder = os.path.join(FLAMBE_GLOBAL_FOLDER, "extensions")
    if os.path.exists(extensions_folder) and get_size_MB(extensions_folder) > WARN_LIMIT_MB:
        print_extensions_cache_size_warning(extensions_folder, WARN_LIMIT_MB)


def _contains_path(nested_dict) -> bool:
    """Whether the nested dict contains any value that could be a path.

    Parameters
    ----------
    nested_dict
        The nested dict to evaluate

    Returns
    -------
    bool

    """
    for v in nested_dict.values():
        if hasattr(v, "values"):
            if _contains_path(v):
                return True
        if isinstance(v, str) and os.sep in v and not _is_url(v):
            return True

    return False


def is_dev_mode() -> bool:
    """Detects if flambe was installed in editable mode.

    For more information:
    https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs

    Returns
    -------
    bool

    """
    x = freeze.freeze()
    for pkg in x:
        if pkg.startswith("-e") and pkg.endswith("egg=flambe"):
            return True

    return False


def get_flambe_repo_location() -> str:
    """Return where flambe repository is located

    Returns
    -------
    str
        The local path where flambe is located

    Raises
    ------
    ValueError
        If flambe was not installed in editable mode

    """
    if not is_dev_mode():
        raise ValueError("Flambe repo can't be located as it was not \
                          installed in editable mode")

    # Go form the top level __init__.py to the flambe repo
    repo_location = os.path.join(flambe.__file__, os.pardir, os.pardir)
    return os.path.abspath(repo_location)


def get_commit_hash() -> str:
    """Get the commit hash of the current flambe development package.

    This will only work if flambe was install from github in dev mode.

    Returns
    -------
    str
        The commit hash

    Raises
    ------
    Exception
        In case flambe was not installed in dev mode.

    """
    x = freeze.freeze()
    for pkg in x:
        if "flambe" in pkg:
            if pkg.startswith("-e"):
                git_url = pkg.split(" ")[-1]
                commit = git_url.split("@")[-1].split("#")[0]
                return commit

    raise Exception("Tried to lookup commit hash in NOT development mode.")
