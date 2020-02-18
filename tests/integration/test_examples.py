import pytest
from flambe.compile import Schema
from flambe.experiment import Experiment
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


def _preprocess_experiment(fname):
    content = list(yaml.load_config_from_file(fname))
    if len(content) == 0:
        return None

    experiment = content[-1]
    if isinstance(experiment, Experiment):
        _reduce_iterations(experiment.pipeline)
        return content

    return None


def run_experiments(base, **kwargs):
    """Run all experiments found in base param.
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
                new_exp = _preprocess_experiment(t.name)
                if new_exp:
                    yaml.dump_config(new_exp, f)
                    print(f.name)
                    ret = subprocess.run(['flambe', 'run', f.name, '-o', d])
                    assert ret.returncode == 0


@pytest.mark.end2end
def test_end2end_experiments(top_level):
    """Runs all experiments found in the integration's
    folder

    """
    tests_base = os.path.dirname(os.path.dirname(__file__))
    base = os.path.join(tests_base, "integration", "end2end")
    run_experiments(base, top_level=top_level)


@pytest.mark.examples
def test_examples_experiments():
    """Runs all experiments found in top level examples
    folder

    """
    tests_base = os.path.dirname(os.path.dirname(__file__))
    base = os.path.join(os.path.dirname(tests_base), "examples")
    run_experiments(base)
