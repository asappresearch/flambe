from typing import Any, Dict, Optional, List, Callable
import sys
import runpy
import tempfile
from copy import deepcopy
from ruamel.yaml import YAML

from flambe.logging import TrialLogging
from flambe.compile import Component
from flambe.runner import Environment


class Script(Component):
    """Implement a Script computable.

    This object can be used to turn any script into a Flambé runnable.
    This is useful when you want to keep your code unchanged. Note
    however that this runnable does not enable checkpointing or
    linking to internal components as it does not have any attributes.

    To use this object, your script needs to be in a pip installable,
    containing all dependencies. The script is run with the following
    command:

    .. code-block:: bash

        python -m script.py --arg1 value1 --arg2 value2

    """

    def __init__(self,
                 script: str,
                 args: List[Any],
                 kwargs: Optional[Dict[str, Any]] = None,
                 pass_env: bool = True,
                 max_steps: Optional[int] = None,
                 metric_fn: Optional[Callable[[str], float]] = None) -> None:
        """Initialize a Script.

        Parameters
        ----------
        path: str
            The script module
        args: List[Any]
            Argument List
        kwargs: Optional[Dict[str, Any]]
            Keyword argument dictionary

        """
        self.script = script
        self.args = args
        if kwargs is None:
            self.kwargs: Dict[str, Any] = {}
        else:
            self.kwargs = kwargs

        self.pass_env = pass_env
        self.max_steps = max_steps
        self.metric_fn = metric_fn
        self._step = 0

        self.register_attrs('_step')

    def metric(self, env: Optional[Environment] = None) -> float:
        """Override to read a metric from your script's output."""
        env = env if env is not None else Environment()
        if self.metric_fn is not None:
            return self.metric_fn(env.output_path)
        return 0.0

    def step(self, env: Optional[Environment] = None) -> bool:
        """Run the evaluation.

        Returns
        -------
        bool
            Whether to continue execution.

        """
        env = env if env is not None else Environment()

        yaml = YAML()
        with tempfile.NamedTemporaryFile() as fp:
            yaml.dump(env, fp)
            # Add extra parameters
            if self.pass_env:
                kwargs = dict(self.kwargs)
                kwargs['--env'] = env

            # Flatten the arguments into a list to pass to sys.argv
            parser_args_flat = [str(item) for item in self.args]
            parser_args_flat += [str(item) for items in kwargs.items() for item in items]

            # Execute the script
            sys_save = deepcopy(sys.argv)
            sys.argv = [''] + parser_args_flat  # add dummy sys[0]
            runpy.run_module(self.script, run_name='__main__', alter_sys=True)
            sys.argv = sys_save

        self._step += 1
        continue_ = True
        if self.max_steps is None or self._step >= self.max_steps:
            continue_ = False
        return continue_

    def run(self, env: Optional[Environment] = None) -> None:
        """Execute the trainer as a Runnable.

        Parameters
        ----------
        env : Environment
            An execution envrionment.

        """
        env = env if env is not None else Environment()
        with TrialLogging(env.output_path, verbose=env.debug):
            continue_ = True
            while continue_:
                continue_ = self.step(env)
