import pytest
from ruamel.yaml.compat import StringIO

from flambe.compile.yaml import load_config, dump_config, Registrable, YAMLLoadType
from flambe.compile.schema import Schema

class A:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
    @classmethod
    def yaml_load_type(cls) -> str:
        return "schematic"

class B:
    def __init__(self, *, x=None, y=None):
        self.x = x
        self.y = y
    @classmethod
    def yaml_load_type(cls) -> str:
        return YAMLLoadType.KWARGS

class C:
    def __init__(self, *args, **kwargs):
        if len(kwargs) == 0:
            assert len(args) <= 1
            self.x = args[0] if len(args) > 0 else None
            self.y = None
        else:
            assert len(args) == 0
            self.x = kwargs['x']
            self.y = kwargs['y'] if 'y' in kwargs else None
    @classmethod
    def yaml_load_type(cls) -> str:
        return "kwargs_or_arg"

class D:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
    @classmethod
    def yaml_load_type(cls) -> str:
        return "kwargs_or_posargs"


def load_one_config(config):
    return list(load_config(config))[0]

def dump_one_config(obj):
    return dump_config([obj])


class TestSchematic:

    def test_empty_args_load(self):
        config = '!tests.unit.compile.test_yaml.A\n'
        a_schema = load_one_config(config)
        assert isinstance(a_schema, Schema)
        assert len(a_schema.arguments) == 2
        print(a_schema.arguments)
        assert all([x is None for x in a_schema.arguments.values()])

    def test_kwargs_load(self):
        config = '!tests.unit.compile.test_yaml.A {x: 5}\n'
        a_schema = load_one_config(config)
        assert isinstance(a_schema, Schema)
        assert len(a_schema.arguments) == 2
        assert a_schema['x'] == 5
        a = a_schema()
        assert a.x == 5
        assert a.y is None

    def test_empty_args_dump(self):
        config = '!tests.unit.compile.test_yaml.A\n'
        a_schema = load_one_config(config)
        expected_config = '!tests.unit.compile.test_yaml.A\nx:\ny:\n'
        assert expected_config == dump_one_config(a_schema)

    def test_nested_dump_two_trip(self):
        config = ("!tests.unit.compile.test_yaml.A\n"
                  "x: !tests.unit.compile.test_yaml.A\n"
                  "y:\n")
        expected_config = ("!tests.unit.compile.test_yaml.A\n"
                  "x: !tests.unit.compile.test_yaml.A\n"
                  "  x:\n"
                  "  y:\n"
                  "y:\n")
        b = load_one_config(config)
        dumped = dump_one_config(b)
        b = load_one_config(dumped)
        dumped = dump_one_config(b)
        assert expected_config == dumped


class TestKwargs:

    def test_empty_args(self):
        config = '!tests.unit.compile.test_yaml.B'
        b = load_one_config(config)
        assert isinstance(b, B)

    def test_empty_args_dump(self):
        config = '!tests.unit.compile.test_yaml.B'
        b = load_one_config(config)
        dumped = dump_one_config(b)
        expected_config = '!tests.unit.compile.test_yaml.B {}\n'
        assert expected_config == dumped

    def test_nested_dump_two_trip(self):
        config = ("!tests.unit.compile.test_yaml.B\n"
                  "x: !tests.unit.compile.test_yaml.B\n"
                  "y:\n")
        expected_config = ("!tests.unit.compile.test_yaml.B\n"
                  "x: !tests.unit.compile.test_yaml.B {}\n"
                  "y:\n")
        b = load_one_config(config)
        dumped = dump_one_config(b)
        b = load_one_config(dumped)
        dumped = dump_one_config(b)
        assert expected_config == dumped


class TestKwargsOrArg:

    def test_empty_args(self):
        config = '!tests.unit.compile.test_yaml.C\n'
        c = load_one_config(config)
        assert isinstance(c, C)

    def test_empty_args_dump(self):
        config = '!tests.unit.compile.test_yaml.C\n'
        c = load_one_config(config)
        dumped = dump_one_config(c)
        expected_config = '!tests.unit.compile.test_yaml.C {}\n'
        assert expected_config == dumped

    def test_nested_dump_two_trip(self):
        config = ("!tests.unit.compile.test_yaml.C\n"
                  "x: !tests.unit.compile.test_yaml.C\n"
                  "y:\n")
        expected_config = ("!tests.unit.compile.test_yaml.C\n"
                  "x: !tests.unit.compile.test_yaml.C {}\n"
                  "y:\n")
        c = load_one_config(config)
        dumped = dump_one_config(c)
        c = load_one_config(dumped)
        dumped = dump_one_config(c)
        assert expected_config == dumped

class TestKwargsOrPosArgs:

    def test_empty_args(self):
        config = '!tests.unit.compile.test_yaml.D\n'
        d = load_one_config(config)
        # assert False
        assert isinstance(d, D)

    def test_empty_args_dump(self):
        config = '!tests.unit.compile.test_yaml.D\n'
        d = load_one_config(config)
        dumped = dump_one_config(d)
        expected_config = '!tests.unit.compile.test_yaml.D {}\n'
        assert expected_config == dumped

    def test_nested_dump_two_trip(self):
        config = ("!tests.unit.compile.test_yaml.D\n"
                  "x: !tests.unit.compile.test_yaml.D\n"
                  "y:\n")
        expected_config = ("!tests.unit.compile.test_yaml.D\n"
                  "x: !tests.unit.compile.test_yaml.D {}\n"
                  "y:\n")
        d = load_one_config(config)
        dumped = dump_one_config(d)
        d = load_one_config(dumped)
        dumped = dump_one_config(d)
        assert expected_config == dumped
