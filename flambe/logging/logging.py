import sys
import os
import pathlib
import logging
from logging import handlers
from typing import Type, Any, List, AnyStr, Optional, Dict  # noqa: F401
from types import TracebackType

from tqdm import tqdm

from flambe.logging.handler.tensorboard import TensorboardXHandler
from flambe.logging.handler.contextual_file import ContextualFileHandler
from flambe.logging.datatypes import DataLoggingFilter
from flambe.const import FLAMBE_GLOBAL_FOLDER

MB = 2**20


def setup_global_logging(console_log_level: int = logging.NOTSET) -> None:
    """Set up flambe logging with a Stream handler and a
    Rotating File handler.

    This method should be set before consuming any logger as it
    sets the basic configuration for all future logs.

    After executing this method, all loggers will have the following
    handlers:
    * Stream handler: prints to std output all logs that above The
    console_log_level
    * Rotating File hanlder: 10MB log file located in Flambe global
    folder. Configured to store all logs (min level DEBUG)

    Parameters
    ----------
    console_log_level: int
        The minimum log level for the Stream handler

    """
    colorize_exceptions()
    logs_dir = os.path.join(FLAMBE_GLOBAL_FOLDER, 'logs')
    pathlib.Path(logs_dir).mkdir(parents=True, exist_ok=True)

    fh = handlers.RotatingFileHandler(
        os.path.join(FLAMBE_GLOBAL_FOLDER, 'logs', 'log.log'),
        maxBytes=10 * MB, backupCount=5
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-8s | %(lineno)04d | %(message)s'
    )
    fh.setFormatter(formatter)

    tqdm_safe_out, tqdm_safe_err = map(TqdmFileWrapper, [sys.stdout, sys.stderr])
    ch = logging.StreamHandler(stream=tqdm_safe_out)  # type: ignore
    formatter = logging.Formatter('%(asctime)s | %(message)s', "%H:%M:%S")
    # Only flambe logs in stdout
    ch.addFilter(FlambeFilter())
    ch.setLevel(console_log_level)
    ch.setFormatter(formatter)

    logging.captureWarnings(True)
    logging.basicConfig(level=logging.DEBUG, handlers=[fh, ch])


class FlambeFilter(logging.Filter):
    """Filter all log records that don't come from flambe or main.

    """
    def filter(self, record: logging.LogRecord) -> bool:
        n = record.name
        return n.startswith("flambe") or n.startswith("__main__")


class TrialLogging:

    def __init__(self,
                 log_dir: str,
                 verbose: bool = False,
                 root_log_level: Optional[int] = None,
                 capture_warnings: bool = True,
                 console_prefix: Optional[str] = None,
                 hyper_params: Optional[Dict] = None) -> None:
        self.log_dir = log_dir
        self.verbose = verbose
        self.log_level = logging.NOTSET
        self.capture_warnings = capture_warnings
        self.listener: handlers.QueueListener
        self.console_prefix = console_prefix
        self.handlers: List[logging.Handler] = []
        self.queue_handler: handlers.QueueHandler
        self.old_root_log_level: int = logging.NOTSET
        self.hyper_params: Dict = hyper_params or {}

    def __enter__(self) -> logging.Logger:
        colorize_exceptions()
        logger = logging.root
        self.old_root_log_level = logger.level
        if self.log_level is not None:
            logger.setLevel(self.log_level)
        console_log_level = logging.NOTSET if self.verbose else logging.ERROR
        console_data_log_level = logging.NOTSET if self.verbose else logging.ERROR
        console_file_log_level = logging.NOTSET if self.verbose else logging.INFO
        tensorboard_log_level = logging.NOTSET if self.verbose else logging.INFO

        # CONSOLE LOGGING
        tqdm_safe_out, tqdm_safe_err = map(TqdmFileWrapper, [sys.stdout, sys.stderr])
        console = logging.StreamHandler(stream=tqdm_safe_out)  # type: ignore
        console.setLevel(console_log_level)
        console_formatter = logging.Formatter('%(name)s [block_%(_console_prefix)s] %(message)s')
        console.setFormatter(console_formatter)
        console_data_filter = DataLoggingFilter(level=console_data_log_level)
        console.addFilter(console_data_filter)
        self.handlers.append(console)

        # RECORD VERBOSE CONSOLE OUTPUT TO CONTEXT-SPECIFIC FILE
        console_splitter = ContextualFileHandler(canonical_name="console.out", mode='a')
        console_splitter.setLevel(console_file_log_level)
        console_splitter.setFormatter(console_formatter)
        self.handlers.append(console_splitter)

        # TENSORBOARDX LOGGING
        try:
            tbx = TensorboardXHandler()
            tbx.setLevel(tensorboard_log_level)
            self.handlers.append(tbx)
        except ModuleNotFoundError:
            print("TensorboardX not found. Disabling logging handler.")

        for handler in self.handlers:
            logger.addHandler(handler)

        # Route built-in Python warnings through our logger, defaulting
        # to WARN severity This means warnings that come from 3rd party
        # code e.g. Pandas, custom user code can be properly filtered
        # and logged
        logging.captureWarnings(self.capture_warnings)

        self.old_factory = logging.getLogRecordFactory()

        def record_factory(name, level, fn, lno, msg, args, exc_info,  # type: ignore
                           func=None, sinfo=None, **kwargs):
            record = self.old_factory(name, level, fn, lno, msg, args,
                                      exc_info, func=None, sinfo=None, **kwargs)
            # Always make raw message available by default so that
            # handlers can manipulate native objects instead of
            # string representations
            record.raw_msg_obj = msg  # type: ignore
            return record

        logging.setLogRecordFactory(record_factory)
        logging.root._log_dir = self.log_dir  # type: ignore

        self.context_filter = ContextInjection(_console_prefix=self.console_prefix,
                                               _tf_log_dir=self.log_dir,
                                               _tf_hparams=self.hyper_params,
                                               _console_log_dir=self.log_dir)

        for handler in logger.handlers:
            handler.addFilter(self.context_filter)
        logger.addFilter(self.context_filter)

        self.logger = logger
        return logger

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the listener and restore original logging config"""
        for handler in self.logger.handlers:
            handler.removeFilter(self.context_filter)
        self.logger.removeFilter(self.context_filter)
        for handler in self.handlers:
            self.logger.removeHandler(handler)
        self.logger.setLevel(self.old_root_log_level)
        logging.setLogRecordFactory(self.old_factory)
        delattr(logging.root, '_log_dir')


class ContextInjection:
    """Add specified attributes to all log records

    Parameters
    ----------
    **attrs : Any
        Attributes that should be added to all log records, for use
        in downstream handlers

    """

    def __init__(self, **attrs) -> None:
        self.attrs = attrs

    def filter(self, record: logging.LogRecord) -> int:
        for k, v in self.attrs.items():
            setattr(record, k, v)
        return True

    def __call__(self, record: logging.LogRecord) -> int:
        return self.filter(record)


class TqdmFileWrapper:
    """Dummy file-like that will write to tqdm

    Based on canoncial tqdm example

    """

    def __init__(self, file: Any) -> None:
        self.file = file

    def write(self, x: AnyStr) -> int:
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            return tqdm.write(x, file=self.file)
        return 0

    def flush(self) -> Any:
        return getattr(self.file, "flush", lambda: None)()


def colorize_exceptions() -> None:
    """Colorizes the system stderr ouput using pygments if installed"""
    try:
        import traceback
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import TerminalFormatter

        def colorized_excepthook(type_: Type[BaseException],
                                 value: BaseException,
                                 tb: TracebackType) -> None:
            tbtext = ''.join(traceback.format_exception(type_, value, tb))
            lexer = get_lexer_by_name("pytb", stripall=True)
            formatter = TerminalFormatter()
            sys.stderr.write(highlight(tbtext, lexer, formatter))

        sys.excepthook = colorized_excepthook  # type: ignore

    except ModuleNotFoundError:
        pass
