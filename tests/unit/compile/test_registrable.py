import pytest

from flambe.compile import yaml, Registrable, alias, register, registrable_factory, \
                                registration_context
from ruamel.yaml.compat import StringIO


@pytest.fixture
def make_classes():

    class A(Registrable):

        def __init__(self, akw1=0, akw2=None):
            self.akw1 = akw1
            self.akw2 = akw2

        @registrable_factory
        @classmethod
        def some_factory(cls, akw1=0, akw2=None):
            return cls(akw1, akw2)

        @classmethod
        def from_yaml(cls, constructor, node, factory_name):
            kwargs, = list(constructor.construct_yaml_map(node))
            if factory_name is not None:
                return getattr(cls, factory_name)(**kwargs)
            else:
                return cls(**kwargs)

        @classmethod
        def to_yaml(cls, representer, node, tag):
            return representer.represent_mapping(tag, {"akw1": node.akw1, "akw2": node.akw2})

    class B(Registrable):

        def __init__(self, bkw1=0, bkw2=''):
            self.bkw1 = bkw1
            self.bkw2 = bkw2

        @classmethod
        def from_yaml(cls, constructor, node, factory_name):
            kwargs, = list(constructor.construct_yaml_map(node))
            return cls(**kwargs)

        @classmethod
        def to_yaml(cls, representer, node, tag):
            return representer.represent_mapping(tag, {"bkw1": node.bkw1, "bkw2": node.bkw2})

    return A, B


@pytest.fixture
def make_namespace_classes():

    with registration_context("ns"):
        class A(Registrable):

            def __init__(self, akw1=0, akw2=None):
                self.akw1 = akw1
                self.akw2 = akw2

            @registrable_factory
            @classmethod
            def some_factory(cls, akw1=0, akw2=None):
                return cls(akw1, akw2)

            @classmethod
            def from_yaml(cls, constructor, node, factory_name):
                kwargs, = list(constructor.construct_yaml_map(node))
                if factory_name is not None:
                    return getattr(cls, factory_name)(**kwargs)
                else:
                    return cls(**kwargs)

            @classmethod
            def to_yaml(cls, representer, node, tag):
                return representer.represent_mapping(tag, {"akw1": node.akw1, "akw2": node.akw2})

        class B(Registrable):

            def __init__(self, bkw1=0, bkw2=''):
                self.bkw1 = bkw1
                self.bkw2 = bkw2

            @classmethod
            def from_yaml(cls, constructor, node, factory_name):
                kwargs, = list(constructor.construct_yaml_map(node))
                return cls(**kwargs)

            @classmethod
            def to_yaml(cls, representer, node, tag):
                return representer.represent_mapping(tag, {"bkw1": node.bkw1, "bkw2": node.bkw2})

    return A, B


@pytest.fixture
def make_aliased_classes():

    @alias('a_class')
    class A(Registrable):

        def __init__(self, akw1=0, akw2=None):
            self.akw1 = akw1
            self.akw2 = akw2

        @registrable_factory
        @classmethod
        def some_factory(cls, akw1=0, akw2=None):
            return cls(akw1, akw2)

        @classmethod
        def from_yaml(cls, constructor, node, factory_name):
            kwargs, = list(constructor.construct_yaml_map(node))
            if factory_name is not None:
                return getattr(cls, factory_name)(**kwargs)
            else:
                return cls(**kwargs)

        @classmethod
        def to_yaml(cls, representer, node, tag):
            return representer.represent_mapping(tag, {"akw1": node.akw1, "akw2": node.akw2})

    @alias('b_class')
    @alias('b_')
    class B(Registrable):

        def __init__(self, bkw1=0, bkw2=''):
            self.bkw1 = bkw1
            self.bkw2 = bkw2

        @classmethod
        def from_yaml(cls, constructor, node, factory_name):
            kwargs, = list(constructor.construct_yaml_map(node))
            return cls(**kwargs)

        @classmethod
        def to_yaml(cls, representer, node, tag):
            return representer.represent_mapping(tag, {"bkw1": node.bkw1, "bkw2": node.bkw2})

    return A, B


@pytest.fixture
def make_new_classes():

    class A:

        def __init__(self, akw1=0, akw2=None):
            self.akw1 = akw1
            self.akw2 = akw2

        @classmethod
        def from_yaml(cls, constructor, node, factory_name):
            kwargs, = list(constructor.construct_yaml_map(node))
            return cls(**kwargs)

        @classmethod
        def to_yaml(cls, representer, node, tag):
            return representer.represent_mapping(tag, {"akw1": node.akw1, "akw2": node.akw2})

    class B:

        def __init__(self, bkw1=0, bkw2=''):
            self.bkw1 = bkw1
            self.bkw2 = bkw2

        @classmethod
        def from_yaml(cls, constructor, node, factory_name):
            kwargs, = list(constructor.construct_yaml_map(node))
            return cls(**kwargs)

        @classmethod
        def to_yaml(cls, representer, node, tag):
            return representer.represent_mapping(tag, {"bkw1": node.bkw1, "bkw2": node.bkw2})

    register(A, 'a_class')
    register(B, 'b_class')
    return A, B


def test_registrable_load_basic(make_classes):
    A, B = make_classes

    txt = """a: !A
  akw1: 8
  akw2: !B
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    a = config['a']
    assert a.akw1 == 8
    assert a.akw2 is not None
    assert hasattr(a.akw2, "bkw1")
    assert a.akw2.bkw1 == 2


def test_registrable_load_context(make_namespace_classes):
    A, B = make_namespace_classes

    txt = """a: !ns.A
  akw1: 8
  akw2: !ns.B
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    a = config['a']
    assert a.akw1 == 8
    assert a.akw2 is not None
    assert hasattr(a.akw2, "bkw1")
    assert a.akw2.bkw1 == 2


def test_registrable_dump_basic(make_classes):
    A, B = make_classes

    txt = """!A
akw1: 8
akw2: !B
  bkw1: 2
  bkw2: hello world
"""

    b = B(2, "hello world")
    a = A(8, b)
    with StringIO() as s:
        yaml.dump(a, s)
        assert s.getvalue() == txt


def test_registrable_roundtrip(make_classes):
    A, B = make_classes

    txt = """a: !A
  akw1: 8
  akw2: !B
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    with StringIO() as s:
        yaml.dump(config, s)
        assert s.getvalue() == txt


def test_registrable_load_alias(make_aliased_classes):
    A, B = make_aliased_classes

    txt = """a: !a_class
  akw1: 8
  akw2: !b_class
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    a = config['a']
    assert a.akw1 == 8
    assert a.akw2 is not None
    assert hasattr(a.akw2, "bkw1")
    assert a.akw2.bkw1 == 2


def test_registrable_dump_alias(make_aliased_classes):
    A, B = make_aliased_classes

    txt = """!a_class
akw1: 8
akw2: !b_class
  bkw1: 2
  bkw2: hello world
"""

    b = B(2, "hello world")
    a = A(8, b)
    with StringIO() as s:
        yaml.dump(a, s)
        assert s.getvalue() == txt


def test_registrable_roundtrip_alias_default(make_aliased_classes):
    A, B = make_aliased_classes

    txt = """a: !a_class
  akw1: 8
  akw2: !b_
    bkw1: 2
    bkw2: hello world
"""
    txt_default_alias = """a: !a_class
  akw1: 8
  akw2: !b_
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    with StringIO() as s:
        yaml.dump(config, s)
        assert s.getvalue() == txt_default_alias


def test_registrable_load_new_class(make_new_classes):
    A, B = make_new_classes

    txt = """a: !a_class
  akw1: 8
  akw2: !b_class
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    a = config['a']
    assert a.akw1 == 8
    assert a.akw2 is not None
    assert hasattr(a.akw2, "bkw1")
    assert a.akw2.bkw1 == 2


def test_registrable_dump_new_class(make_new_classes):
    A, B = make_new_classes

    txt = """!a_class
akw1: 8
akw2: !b_class
  bkw1: 2
  bkw2: hello world
"""

    b = B(2, "hello world")
    a = A(8, b)
    with StringIO() as s:
        yaml.dump(a, s)
        assert s.getvalue() == txt


def test_registrable_roundtrip_new_default(make_new_classes):
    A, B = make_new_classes

    txt = """a: !a_class
  akw1: 8
  akw2: !b_
    bkw1: 2
    bkw2: hello world
"""
    txt_default_alias = """a: !a_class
  akw1: 8
  akw2: !b_
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    with StringIO() as s:
        yaml.dump(config, s)
        assert s.getvalue() == txt_default_alias


def test_registrable_factory(make_classes):
    A, B = make_classes

    txt = """a: !A.some_factory
  akw1: 8
  akw2: !B
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    a = config['a']
    assert a.akw1 == 8
    assert a.akw2 is not None
    assert hasattr(a.akw2, "bkw1")
    assert a.akw2.bkw1 == 2


def test_registrable_factory_roundtrip(make_classes):
    A, B = make_classes

    txt = """a: !A.some_factory
  akw1: 8
  akw2: !B
    bkw1: 2
    bkw2: hello world
"""
    txt_default_alias = """a: !A.some_factory
  akw1: 8
  akw2: !B
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    with StringIO() as s:
        yaml.dump(config, s)
        assert s.getvalue() == txt_default_alias


def test_registrable_factory_roundtrip_alias(make_aliased_classes):
    A, B = make_aliased_classes

    txt = """a: !a_class.some_factory
  akw1: 8
  akw2: !b_
    bkw1: 2
    bkw2: hello world
"""
    txt_default_alias = """a: !a_class.some_factory
  akw1: 8
  akw2: !b_
    bkw1: 2
    bkw2: hello world
"""
    config = yaml.load(txt)
    a = config['a']
    assert a.akw1 == 8
    assert a.akw2 is not None
    assert hasattr(a.akw2, "bkw1")
    assert a.akw2.bkw1 == 2
    assert isinstance(a, A)
    with StringIO() as s:
        yaml.dump(config, s)
        assert s.getvalue() == txt_default_alias
