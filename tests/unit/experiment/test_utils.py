import pytest
import ray

from flambe.compile import Component, yaml
from flambe.experiment.utils import divide_nested_grid_search_options, get_default_devices


@pytest.fixture
def make_classes():
    class A(Component):
        def __init__(self, akw1=0, akw2=None):
            self.akw1 = akw1
            self.akw2 = akw2

    class B(Component):
        def __init__(self, bkw1=0, bkw2=''):
            self.bkw1 = bkw1
            self.bkw2 = bkw2
    return A, B


def test_divide_nested_grid_search_options_no_options(make_classes):
    A, B = make_classes
    txt = """
!A
akw1: 8
akw2: !B
  bkw1: 1
  bkw2: 'hello'
"""
    config = yaml.load(txt)
    divided_configs = list(divide_nested_grid_search_options(config))
    assert repr(divided_configs) == repr([config])


def test_divide_nested_grid_search_options_non_nested_options(make_classes):
    A, B = make_classes
    txt = """
!A
akw1: 8
akw2: !B
  bkw1: !g [1, 3, 5]
  bkw2: 'hello'
"""
    config = yaml.load(txt)
    divided_configs = list(divide_nested_grid_search_options(config))
    assert repr(divided_configs) == repr([config])


def test_divide_nested_grid_search_options_nested_schemas(make_classes):
    A, B = make_classes
    txt = """
!A
akw1: 8
akw2: !g
  - !B
    bkw1: 2
    bkw2: 'first'
  - !B
    bkw1: 3
    bkw2: 'second'
"""
    txt_1 = """
!A
akw1: 8
akw2: !B
  bkw1: 2
  bkw2: 'first'
"""
    txt_2 = """
!A
akw1: 8
akw2: !B
  bkw1: 3
  bkw2: 'second'
"""
    config = yaml.load(txt)
    config1 = yaml.load(txt_1)
    config2 = yaml.load(txt_2)
    divided_configs = list(divide_nested_grid_search_options(config))
    assert repr(divided_configs) == repr([config1, config2])


def test_divide_nested_grid_search_options_nested_options(make_classes):
    A, B = make_classes
    txt = """
!A
akw1: 8
akw2: !g
  - !B
    bkw1: !g [4, 5]
    bkw2: 'first'
  - !B
    bkw1: 3
    bkw2: !g ['second', 'third']
"""
    txt_1 = """
!A
akw1: 8
akw2: !B
  bkw1: !g [4, 5]
  bkw2: 'first'
"""
    txt_2 = """
!A
akw1: 8
akw2: !B
  bkw1: 3
  bkw2: !g ['second', 'third']
"""
    config = yaml.load(txt)
    config1 = yaml.load(txt_1)
    config2 = yaml.load(txt_2)
    divided_configs = list(divide_nested_grid_search_options(config))
    assert repr(divided_configs) == repr([config1, config2])


def test_default_devices():
    ray.init()
    devices = get_default_devices(debug=False)
    assert devices == {'cpu': 1}

    devices = get_default_devices(debug, default_cpus=2)
    assert devices == {'cpu': 2}

    devices = get_default_devices(debug, default_gpus=2)
    assert devices == {'cpu': 1}


def test_default_devices_debug():
    devices = get_default_devices(debug=True)
    assert devices == {'cpu': 1}

    devices = get_default_devices(debug, default_cpus=2)
    assert devices == {'cpu': 2}

    devices = get_default_devices(debug, default_gpus=2)
    assert devices == {'cpu': 1}