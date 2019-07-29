import colorama
from colorama import Fore, Style
import logging
colorama.init()

logger = logging.getLogger(__name__)


def print_useful_local_info(full_save_path) -> None:
    """Information to display before experiment is running.

    """
    logger.info("###########################################################")
    logger.info("                     IMPORTANT")
    logger.info("###########################################################")
    logger.info("")
    logger.info("Experiment will start running shortly. Results will start to become available")
    logger.info(f"at {full_save_path}.")
    logger.info("")
    logger.info("REPORT SITE")
    logger.info("----------")
    logger.info("To view the experiment's progress, run:")
    logger.info("")
    logger.info(f"$ flambe-site {full_save_path}/state.pkl [--port 12345]")
    logger.info("")


def print_useful_remote_info(manager, experiment_name) -> None:
    """ Once the local process of the remote run is over,
    this information is shown to the user.

    """
    logger.info("###########################################################")
    logger.info("                     IMPORTANT")
    logger.info("###########################################################")
    logger.info("")
    logger.info("MODEL DOWNLOADING")
    logger.info("-----------------")
    logger.info("Once experiment is over, you can download all results by executing:")
    logger.info(
        f"scp -r -i <key> {manager.orchestrator.username}@{manager.orchestrator.host}:"
        f"{manager.orchestrator.get_home_path()}/{experiment_name}/synced_results <local_dst>")
    logger.info("")
    logger.info("Where <key> is the private key used to create the cluster and <local_dst>")
    logger.info("is a local directory to store the results.")

    logger.info("")
    logger.info("INSTANCES")
    logger.info("---------")

    if manager.factories_timeout > 0:
        logger.info(
            f"The factories instances  will be terminated after {manager.factories_timeout} " +
            "hour of unusage.")
    elif manager.factories_timeout == -1:
        logger.info("The factories instances  will remain running " +
                    "(you are in charge of terminating them)")
    elif manager.factories_timeout == 0:
        logger.info("The factories instance  will be terminated once the experiment is over")

    if manager.orchestrator_timeout > 0:
        logger.info(f"The orchestrator will be terminated after {manager.orchestrator_timeout} " +
                    "hour of unusage.")
    elif manager.orchestrator_timeout == -1:
        logger.info("The orchestrator instance  will remain running " +
                    "(you are in charge of terminating it)")
    elif manager.orchestrator_timeout == 0:
        logger.info("The orchestrator instance  will be terminated once the experiment is over")


def print_useful_metrics_only_info() -> None:
    """Printable warning when debug flag is active

    """
    logger.info(Fore.YELLOW)
    logger.info("You specified 'debug' to True and therefore, after experiment is done, " +
                "all data will be lost!")
    logger.info(Style.RESET_ALL)


def print_extensions_cache_size_warning(location, limit) -> None:
    """Print message when the extensions cache folder is getting big.

    """
    logger.info(Fore.YELLOW)
    logger.info(
        f"Be aware that your extensions cache for github extensions located in {location}) " +
        f"is increasing its size (it's currently bigger than {limit} MB).")
    logger.info("Please remove unused extensions from that location.")
    logger.info(Style.RESET_ALL)
