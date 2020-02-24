import pytest
from flambe.compile import Schema
from flambe.pipeline import Pipeline
from tempfile import TemporaryDirectory as tmpdir
from tempfile import NamedTemporaryFile as tmpfile
import subprocess
import os
from flambe.compile import yaml


def _reduce_iterations(d):
    """Recursively update any iteration's config

    """
    for k, v in d.items():
        if k == 'max_steps' or k == 'iter_per_step' or k == 'epoch_per_step':
            d[k] = 1

        elif isinstance(v, Schema):
            _reduce_iterations(v)


def run_tasks(base, **kwargs):
    """Run all tasks found in base param.
    and check that flambe executes without errors.

    Before running the configs, it updates the save_path to
    be a tempdir and updates (potentially) the iteration's
    params (if found) to be 1.

    """
    for fname in os.listdir(base):
        full_f = os.path.join(base, fname)
        if os.path.isfile(full_f) and fname.endswith('yaml'):
            with tmpdir() as d, tmpfile() as f, tmpfile('w') as t:
                content = open(full_f).read().format(**kwargs)
                t.write(content)
                t.flush()
                ret = subprocess.run(['flambe', 'run', t.name, '-o', d])
                assert ret.returncode == 0


@pytest.mark.end2end
def test_end2end_tasks(top_level):
    """Runs all tasks found in the integration's
    folder

    """
    tests_base = os.path.dirname(os.path.dirname(__file__))
    base = os.path.join(tests_base, "integration", "end2end")
    run_tasks(base, top_level=top_level)


@pytest.mark.examples
def test_examples_tasks():
    """Runs all tasks found in top level examples
    folder

    """
    tests_base = os.path.dirname(os.path.dirname(__file__))
    base = os.path.join(os.path.dirname(tests_base), "examples")
    run_tasks(base)
