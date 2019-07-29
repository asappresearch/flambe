import logging
from typing import Any, Dict

from tensorboardX import SummaryWriter

from flambe.logging.datatypes import ScalarT, ScalarsT, HistogramT, TextT, \
    ImageT, EmbeddingT, PRCurveT, GraphT, DataLoggingFilter


class TensorboardXHandler(logging.Handler):
    """Implements Tensorboard message logging via TensorboardX

    Parameters
    ----------
    writer : SummaryWriter
        Initialized TensorboardX Writer
    *args : Any
        Other positional args for `logging.Handler`
    **kwargs : Any
        Other kwargs for `logging.Handler`

    Attributes
    ----------
    writer : SummaryWriter
        Initialized TensorboardX Writer

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # TODO remove dont_include when we add support for saving graph
        self.addFilter(DataLoggingFilter(default=False, dont_include=(GraphT,)))

        self.writers: Dict[str, SummaryWriter] = {}

    def emit(self, record: logging.LogRecord) -> None:
        """Save to tensorboard logging directory

        Overrides `logging.Handler.emit`

        Parameters
        ----------
        record : logging.LogRecord
            LogRecord with data relevant to Tensorboard

        Returns
        -------
        None

        """
        # Handler relies on access to raw objects which flambe logging
        # provides
        if not hasattr(record, "raw_msg_obj"):
            return
        message = record.raw_msg_obj  # type: ignore
        # Check for a log directory from the logging context
        # This will be prepended to the final tag before saving to
        # Tensorboard

        if hasattr(record, "_tf_log_dir"):
            log_dir = record._tf_log_dir  # type: ignore
            if log_dir in self.writers:
                writer = self.writers[log_dir]
            else:
                writer = SummaryWriter(log_dir=log_dir)
                hparams = getattr(record, "_tf_hparams", dict())
                if len(hparams):
                    writer.add_hparams_start(hparams=hparams)

                self.writers[log_dir] = writer
        else:
            return
        # Datatypes with a standard `tag` field
        if isinstance(message, (ScalarT, HistogramT, TextT, EmbeddingT, ImageT, PRCurveT)):
            kwargs = message._replace(tag=message.tag)._asdict()
            fn = {
                ScalarT: writer.add_scalar,
                HistogramT: writer.add_histogram,
                TextT: writer.add_text,
                EmbeddingT: writer.add_embedding,
                ImageT: writer.add_image,
                PRCurveT: writer.add_pr_curve
            }
            fn[message.__class__](**kwargs)
        # Datatypes with a special tag field
        elif isinstance(message, ScalarsT):
            kwargs = message._replace(main_tag=message.main_tag)._asdict()
            writer.add_scalars(**kwargs)
        # Datatypes without a tag field
        elif isinstance(message, GraphT):
            kwargs = message._asdict()
            for k, v in kwargs['kwargs']:
                kwargs[k] = v
            del kwargs['kwargs']
            writer.add_model(**kwargs)
        writer.file_writer.flush()

    def close(self) -> None:
        """Teardown writers and teardown super

        Returns
        -------
        None

        """
        # Use built-in writer `close` method to flush and close
        for _, w in self.writers.items():
            w.add_hparams_end()
            w.close()

        super().close()

    def flush(self) -> None:
        """Call flush on the Tensorboard writer

        Returns
        -------
        None

        """
        # No public `flush` method is available on the writer, so
        # copy the flushing logic from the TensorboardX `SummaryWriter`

        for _, w in self.writers.items():
            if w.file_writer:
                w.file_writer.flush()

        for _, w in self.writers.items():
            for path, writer in w.all_writers.items():
                writer.flush()

        super().flush()
