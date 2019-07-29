"""Script to run the garbage collector.

The garbage collector is in charge of destroying the factories
once the experiment is over.

All useful information was sent to the orchestrator machine,
so the Factories can be safely deleted.

To do this, the GarbageCollector gets the tasks from the Redis DB
running in the Orchestrator. Once all tasks are over, it gets the
factories from the Redis DB also (that's why the Instances should be
pickeable) and terminates them.

"""
import logging
import argparse

import redis
import pickle

logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    r = redis.Redis.from_url(args.redis_url)
    tasks = pickle.loads(r.get("tasks"))

    for k, t in tasks.items():
        t.get()  # Blocks till process is over

    print("Experiment is over")
    logger.info("Experiment is over")

    # This is why the Instances (especially the Factories)
    # need to be pickeable
    factories = pickle.loads(r.get("factories"))
    for f in factories:
        f.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flambe garbage collector')
    parser.add_argument('redis_url', type=str, default='localhost',
                        help='Redis url.')
    args = parser.parse_args()

    try:
        main(args)
        logger.warning("\n------------------- Done -------------------\n")
    except KeyboardInterrupt:
        logger.warning("\n---- Exiting early (Keyboard Interrupt) ----\n")
