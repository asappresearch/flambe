import pytest

from flambe import Component
from flambe.compile import yaml
from flambe.compile.component import MalformedLinkError, parse_link_str, create_link_str
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
        return yaml.load(config)()

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

    config = yaml.load(txt)
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
    a_schema = yaml.load(txt)
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
    a_schema = yaml.load(txt)
    assert a_schema['akw1'] == 8
    assert a_schema['akw2']['bkw2'] == 'test'
    a_schema['akw2']['bkw1'] = 13
    assert a_schema['akw2']['bkw1'] == 13
    a_schema.keywords['akw2'].keywords['bkw1'] = 14
    a = a_schema()
    assert a.akw2.bkw1 == 14


def test_component_override(make_classes):
    A, B = make_classes

    txt = """
!A
akw1: 8
akw2: !B
  bkw1: 1
  bkw2: 'test'
"""
    a_schema = yaml.load(txt)
    a = a_schema(akw1=9)
    assert a.akw1 == 9


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
    a_schema = yaml.load(txt)
    a = a_schema()
    with StringIO() as stream:
        yaml.dump(a, stream)
        assert txt_expected == stream.getvalue()


def test_component_dumping_factory(make_instances):
    a = make_instances()
    config = "!A {}\n"
    with StringIO() as stream:
        yaml.dump(a, stream)
        assert config == stream.getvalue()


def test_component_dumping_made_in_code(make_classes_2):
    A, B = make_classes_2

    txt_expected = """!A
akw1: 8
akw2: !B
  bkw1: 1
  bkw2: test
  bkw3: 99
"""
    b_custom = B.compile(bkw1=1, bkw2='test')
    a_custom = A.compile(akw1=8, akw2=b_custom)
    with StringIO() as stream:
        yaml.dump(a_custom, stream)
        assert txt_expected == stream.getvalue()


def test_component_anchors_compile_to_same_instance(make_classes_2):

    txt = """
one: !A
  akw2: &theb !B
    bkw2: test
    bkw1: 1
  akw1: 8
two: !A
  akw1: 8
  # Comment Here
  akw2: *theb
"""
    config = yaml.load(txt)
    a1 = config["one"]()
    a1.akw2.bkw1 = 6
    a2 = config["two"]()
    assert a1.akw2 is a2.akw2
    assert a1.akw2.bkw1 == a2.akw2.bkw1


def test_component_linking():
    # TODO test links
    pass


def test_component_dynamic():
    # TODO test dynamically created Components
    pass


def test_state_dict_basic(make_classes):
#     A, B = make_classes
#
#     txt = """
# top: !A
#   akw1: 8
#   akw2: !B
#     bkw1: 1
#     bkw2: 'test'
# """
#     config = yaml.load(txt)
#     a = config['top']()
#     s = a.state_dict()
#     print(s)
#     assert False
    pass


def test_state_dict_roundtrip():
    pass


def test_state_dict_basic_pytorch():
    pass


def test_state_dict_roundtrip_original_source():
    pass


def test_state_dict_roundtrip_new_source():
    pass


def test_save_basic():
    pass


def test_load_basic():
    pass


def test_load_save_roundtrip():
    pass


class TestLinkParser:

    def test_only_obj(self):
        link = 'model'
        assert parse_link_str(link) == (['model'], [])

    def test_only_attr(self):
        link = 'model.emb'
        assert parse_link_str(link) == (['model'], ['emb'])
        link = 'model.emb.enc'
        assert parse_link_str(link) == (['model'], ['emb', 'enc'])

    def test_only_schematic(self):
        link = 'model[emb]'
        assert parse_link_str(link) == (['model', 'emb'], [])
        link = 'model[emb][enc]'
        assert parse_link_str(link) == (['model', 'emb', 'enc'], [])

    def test_schematic_and_attr(self):
        link = 'model[emb].attr1'
        assert parse_link_str(link) == (['model', 'emb'], ['attr1'])
        link = 'model[emb][enc].attr1.attr2'
        assert parse_link_str(link) == (['model', 'emb', 'enc'], ['attr1', 'attr2'])

    def test_close_unopen_schematic(self):
        with pytest.raises(MalformedLinkError):
            link = 'modelemb][enc].attr1.attr2'
            parse_link_str(link)

    def test_close_unopen_schematic_2(self):
        with pytest.raises(MalformedLinkError):
            link = 'model[emb]enc].attr1.attr2'
            parse_link_str(link)

    def test_reopen_schematic(self):
        with pytest.raises(MalformedLinkError):
            link = 'model[emb[enc].attr1.attr2'
            parse_link_str(link)

    def test_attr_without_dot(self):
        with pytest.raises(MalformedLinkError):
            link = 'model[emb][enc]attr1.attr2'
            parse_link_str(link)

    def test_no_root_obj(self):
        with pytest.raises(MalformedLinkError):
            link = '[emb]'
            parse_link_str(link)
        with pytest.raises(MalformedLinkError):
            link = '.attr2'
            parse_link_str(link)
        with pytest.raises(MalformedLinkError):
            link = '[emb][enc].attr1.attr2'
            parse_link_str(link)


class TestLinkCreator:

    def test_only_obj(self):
        link = (['model'], [])
        assert create_link_str(*link) == 'model'

    def test_only_attr(self):
        link = (['model'], ['emb'])
        assert create_link_str(*link) == 'model.emb'
        link = (['model'], ['emb', 'enc'])
        assert create_link_str(*link) == 'model.emb.enc'

    def test_only_schematic(self):
        link = (['model', 'emb'], [])
        assert create_link_str(*link) == 'model[emb]'
        link = (['model', 'emb', 'enc'], [])
        assert create_link_str(*link) == 'model[emb][enc]'

    def test_schematic_and_attr(self):
        link = (['model', 'emb'], ['attr1'])
        assert create_link_str(*link) == 'model[emb].attr1'
        link = (['model', 'emb', 'enc'], ['attr1', 'attr2'])
        assert create_link_str(*link) == 'model[emb][enc].attr1.attr2'
