"""Script to run the report web site

It takes the `app` defined in `flambe.remote.webapp.app` and runs it.

"""
import logging
import argparse
from typing import Optional
import os

from flambe.experiment.webapp.app import app

logger = logging.getLogger(__name__)


def launch_tensorboard(tracking_address) -> Optional[str]:
    # https://stackoverflow.com/a/55708102
    # tb will run in background but it will
    # be stopped once the main process is stopped.
    try:
        from tensorboard import program
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tracking_address])
        url = tb.launch()
        if url.endswith("/"):
            url = url[:-1]

        return url
    except Exception:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flambe webapp')
    parser.add_argument('progress_file', type=str, default='localhost',
                        help='The location of the pickled progress file')
    parser.add_argument('--output-log', type=str, default=None,
                        help='The experiment log file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='The experiment output directory')
    parser.add_argument('--host', type=str, default="localhost",
                        help='Port in which the site will be running url')
    parser.add_argument('--port', type=int, default=49558,
                        help='Port in which the site will be running url')
    parser.add_argument('--tensorboard_url', type=str, help='Tensorboard url')
    args = parser.parse_args()

    app.config['progress_file'] = args.progress_file
    app.config['output_log'] = args.output_log
    app.config['output_dir'] = args.output_dir
    app.config['tensorboard_url'] = args.tensorboard_url

    if app.config['tensorboard_url'] is None:
        app.config['tensorboard_url'] = launch_tensorboard(
            os.path.dirname(app.config['progress_file']))

    # debug=False won't work remotely as Flask will not be able
    # to bind 0.0.0.0. Make sure to use debug=True is the report site
    # runs remotely.
    # Only use False for debugging purposes.
    app.run(host=args.host, port=args.port, debug=False)
