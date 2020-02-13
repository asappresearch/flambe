import pytest

from flambe.compile import load_first_config, dump_one_config
from flambe import Component
from ruamel.yaml.compat import StringIO


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


@pytest.fixture
def make_classes_2():

    class A(Component):

        def __init__(self, akw1=0, akw2=None):
            self.akw1 = akw1
            self.akw2 = akw2

    class B(Component):

        def __init__(self, bkw1=0, bkw2='', bkw3=99):
            self.bkw1 = bkw1
            self.bkw2 = bkw2
            self.bkw3 = bkw3

    return A, B


@pytest.fixture
def make_instances():
    class A(Component):
        pass

    def _factory():
        config = "!A {}\n"
        return load_first_config(config)()

    return _factory


def test_component_basic(make_classes):
    A, B = make_classes

    txt = """
top: !A
  akw1: 8
  akw2: !B
    bkw1: 1
    bkw2: 'test'
"""

    config = load_first_config(txt)
    a = config['top']()
    assert a.akw1 == 8
    assert a.akw2 is not None
    assert a.akw2.bkw1 == 1


def test_component_basic_top_level(make_classes):
    A, B = make_classes

    txt = """
!A
akw1: 8
akw2: !B
  bkw1: 1
  bkw2: 'test'
"""
    a_schema = load_first_config(txt)
    a = a_schema()
    assert a.akw1 == 8
    assert a.akw2 is not None
    assert a.akw2.bkw1 == 1


def test_component_schema_dict_access(make_classes):
    A, B = make_classes

    txt = """
!A
akw1: 8
akw2: !B
  bkw1: 1
  bkw2: 'test'
"""
    a_schema = load_first_config(txt)
    assert a_schema['akw1'] == 8
    assert a_schema['akw2']['bkw2'] == 'test'
    a_schema['akw2']['bkw1'] = 13
    assert a_schema['akw2']['bkw1'] == 13
    a_schema['akw2']['bkw1'] = 14
    a = a_schema()
    assert a.akw2.bkw1 == 14


def test_component_dumping_with_defaults_and_comments(make_classes_2):
    A, B = make_classes_2

    txt = """!A
akw1: 8
# Comment Here
akw2: !B
  bkw1: 1
  bkw2: !!str test
"""
    txt_expected = """!A
akw1: 8
akw2: !B
  bkw1: 1
  bkw2: test
  bkw3: 99
"""
    a_schema = load_first_config(txt)
    with StringIO() as stream:
        dump_one_config(a_schema, stream)
        assert txt_expected == stream.getvalue()


def test_component_dumping_made_in_code(make_classes_2):
    A, B = make_classes_2

    txt_expected = """!A
akw1: 8
akw2: !B
  bkw1: 1
  bkw2: test
  bkw3: 99
"""
    b_custom = B.schema(bkw1=1, bkw2='test')
    a_custom = A.schema(akw1=8, akw2=b_custom)
    with StringIO() as stream:
        dump_one_config(a_custom, stream)
        assert txt_expected == stream.getvalue()
