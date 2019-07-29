from flambe.cluster import AWSCluster

import pytest
import tempfile
import shutil
import sys
from io import StringIO
import importlib
import copy
import configparser

from flambe.runnable import Runnable

@pytest.fixture
def runnable():
    class DummyRunnable(Runnable):
        def run(self, **kwargs) -> None:
            pass

    return DummyRunnable()


@pytest.fixture
def get_secrets():
    t = tempfile.NamedTemporaryFile(mode="w+")

    def _get_secrets(secrets_content):
        t.write(secrets_content)
        t.flush()
        return t.name

    yield _get_secrets
    t.close()


def test_valid_secrets(runnable, get_secrets):
    secrets = """
        [SOME_SECTION]
        RANDOM = random
    """

    runnable.inject_secrets(get_secrets(secrets))


def test_invalid_secrets(runnable, get_secrets):
    secrets = """
        this is a text
    """

    with pytest.raises(configparser.MissingSectionHeaderError):
        runnable.inject_secrets(get_secrets(secrets))


def test_invalid_secrets2(runnable, get_secrets):
    secrets = """
        {json: tru}
    """

    with pytest.raises(configparser.MissingSectionHeaderError):
        runnable.inject_secrets(get_secrets(secrets))


def test_no_secrets(runnable):
    assert len(runnable.config) == 1
    assert list(runnable.config.keys())[0] == 'DEFAULT'


def test_secrets(runnable, get_secrets):
    secrets = """
        [SOME_SECTION]
        RANDOM = random
    """

    runnable.inject_secrets(get_secrets(secrets))

    assert 'SOME_SECTION' in runnable.config
    assert 'RANDOM' in runnable.config['SOME_SECTION']
    assert runnable.config['SOME_SECTION']['RANDOM'] == 'random'
