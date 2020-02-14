import ray

from flambe.runner import Environment


def initialize(env: Environment):
    if not ray.is_initialized():
        if env.remote:
            ray.init("auto", local_mode=env.debug)
        else:
            ray.init(local_mode=env.debug)
