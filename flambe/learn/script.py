from typing import Any, Dict, Optional
import sys
import runpy
from copy import deepcopy

from flambe.logging import get_trial_dir
from flambe.compile import Component


class Script(Component):
    """Implement a Script computable.

    The obejct can be used to turn any script into a Flambé computable.
    This is useful when you want to rapidly integrate code. Note
    however that this computable does not enable checkpointing or
    linking to internal components as it does not have any attributes.

    To use this object, your script needs to be in a pip installable,
    containing all dependencies. The script is run with the following
    command:

    .. code-block:: bash

        python -m script.py --arg1 value1 --arg2 value2

    """

    def __init__(self,
                 script: str,
                 args: Dict[str, Any],
                 output_dir_arg: Optional[str] = None) -> None:
        """Initialize a Script.

        Parameters
        ----------
        path: str
            The script module
        args: Dict[str, Any]
            Argument dictionary
        output_dir_arg: str, optional
            The name of the argument corresponding to the output
            directory, should there be one.

        """
        self.script = script
        self.args = args

        if output_dir_arg is not None:
            self.args[output_dir_arg] = get_trial_dir()

    def run(self) -> bool:
        """Run the evaluation.

        Returns
        -------
        Dict[str, float]
            Report dictionary to use for logging

        """
        parser_args = {f'--{k}': v for k, v in self.args.items()}
        # Flatten the arguments into a single list to pass to sys.argv
        parser_args_flat = [str(item) for items in parser_args.items() for item in items]

        sys_save = deepcopy(sys.argv)
        sys.argv = [''] + parser_args_flat  # add dummy sys[0]
        runpy.run_module(self.script, run_name='__main__', alter_sys=True)
        sys.argv = sys_save

        continue_ = False  # Single step, so don't continue
        return continue_
