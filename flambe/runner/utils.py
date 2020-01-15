import os
import logging
import colorama

try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze

import flambe
from flambe.const import FLAMBE_GLOBAL_FOLDER, MB, WARN_LIMIT_MB


logger = logging.getLogger(__name__)


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
        logger.info(colorama.Fore.YELLOW)
        logger.info(
            f"Be aware that your extensions cache for github extensions \
              located in {extensions_folder} is increasing its size \
              (it's currently bigger than {WARN_LIMIT_MB} MB).")
        logger.info("Please remove unused extensions from that location.")
        logger.info(colorama.Style.RESET_ALL)


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
