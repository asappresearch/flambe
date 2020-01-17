import pytest

from flambe.compile.schema import Schema, Link, MalformedLinkError


class TestSchemaInitialize:

    def test_initialize_creates_instance(self):

        class A:
            pass

        schema = Schema(A)
        a = schema.initialize()
        assert isinstance(a, A)

    def test_initialize_captures_kwargs(self):

        class A:
            def __init__(self, x):
                self.x = x

        kwargs = {'x': 1}
        schema = Schema(A, kwargs=kwargs)
        a = schema.initialize()
        assert a.x == 1

    def test_initialize_captures_args(self):

        class A:
            def __init__(self, x):
                self.x = x

        args = [1]
        schema = Schema(A, args=args)
        a = schema.initialize()
        assert a.x == 1

    def test_initialize_captures_all_args(self):
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        args = [1]
        kwargs = {'y': 2}
        schema = Schema(A, args=args, kwargs=kwargs)
        a = schema.initialize()
        assert a.x == 1
        assert a.y == 2

    def test_initialize_recursive(self):
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        class B:
            def __init__(self, z):
                self.z = z
        sb = Schema(B, kwargs={'z': 2})
        sa = Schema(A, args=[1, sb])
        a = sa.initialize()
        assert a.x == 1
        assert a.y.z == 2

    def test_initialize_recursive_links(self):
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        class B:
            def __init__(self):
                pass
        sb = Schema(B)
        sa = Schema(A, args=[sb, Link(link_str='x')])
        a = sa.initialize()
        assert a.x is a.y

    def test_initialize_fails_non_init_link(self):
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        class B:
            def __init__(self, z):
                self.z = z
        sx = Schema(B, args=[Link(link_str='y')])
        sy = Schema(B, args=[2])
        sa = Schema(A, kwargs={'x':sx, 'y':sy})
        with pytest.raises(MalformedLinkError):
            a = sa.initialize()


class TestSchemaArgs:

    def test_fails_duplicate_args(self):
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        args = [1]
        kwargs = {'x': 2}
        with pytest.raises(TypeError):
            schema = Schema(A, args=args, kwargs=kwargs)

    def test_fails_missing_required_args(self):
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        kwargs = {'x': 2}
        with pytest.raises(TypeError):
            schema = Schema(A, kwargs=kwargs)

    def test_fails_star_args(self):
        class A:
            def __init__(self, *args, y=3, **kwargs):
                if 'x' in kwargs:
                    self.x = kwargs['x']
                self.y = y

        kwargs = {'x': 1}
        with pytest.raises(TypeError):
            schema = Schema(A, kwargs=kwargs)
