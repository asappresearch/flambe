import pytest

from flambe.compile.schema import Schema, Link, CopyLink, MalformedLinkError


class TestSchemaInitialize:

    def test_initialize_creates_instance(self):
        """Test initialization runs for simple schema"""

        class A:
            pass

        schema = Schema(A)
        a = schema.initialize()
        assert isinstance(a, A)

    def test_initialize_captures_kwargs(self):
        """Test initialization captures keyword args"""

        class A:
            def __init__(self, x):
                self.x = x

        kwargs = {'x': 1}
        schema = Schema(A, kwargs=kwargs)
        a = schema.initialize()
        assert a.x == 1

    def test_initialize_captures_args(self):
        """Test initialization captures positional args"""

        class A:
            def __init__(self, x):
                self.x = x

        args = [1]
        schema = Schema(A, args=args)
        a = schema.initialize()
        assert a.x == 1

    def test_initialize_captures_all_args(self):
        """Test initialization captures positional and keyword args"""
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
        """Test initialization will recurse to child schemas"""
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

    def test_initialize_recursive_deep(self):
        """Test initialization will recurse to grandchild schemas"""
        class A:
            def __init__(self, ax, ay):
                self.ax = ax
                self.ay = ay
        class B:
            def __init__(self, bx, by):
                self.bx = bx
                self.by = by
        class C:
            def __init__(self, cx, cy):
                self.cx = cx
                self.cy = cy
        s1 = Schema(C, args=[None, 1])
        s2 = Schema(B, args=[s1, 2])
        s3 = Schema(A, args=[s2, 3])
        cache = {}
        a = s3.initialize(cache=cache)
        assert a.ay == 3
        assert a.ax.by == 2
        assert a.ax.bx.cy == 1
        assert a.ax.bx.cx is None

class TestSchemaArgs:

    def test_fails_duplicate_args(self):
        """Test specifying same argument twice will fail immediately"""
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        args = [1]
        kwargs = {'x': 2}
        with pytest.raises(TypeError):
            schema = Schema(A, args=args, kwargs=kwargs)

    def test_fails_missing_required_args(self):
        """Test missing argument will fail immediately"""
        class A:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        kwargs = {'x': 2}
        with pytest.raises(TypeError):
            schema = Schema(A, kwargs=kwargs)

    def test_fails_star_args(self):
        """Test star args are prohibited"""
        class A:
            def __init__(self, *args, y=3, **kwargs):
                if 'x' in kwargs:
                    self.x = kwargs['x']
                self.y = y

        kwargs = {'x': 1}
        with pytest.raises(TypeError):
            schema = Schema(A, kwargs=kwargs)


# Classes used for link testing below

class G:
    def __init__(self, x, y):
        self.x = x
        self.y = y
class H:
    def __init__(self, z):
        self.z = z

class O:
    def __init__(self, val=-1):
        self.val = val
class T:
    def __init__(self, x):
        self.x = x
class A:
    def __init__(self, t1):
        self.t1 = t1
class B:
    def __init__(self, t2):
        self.t2 = t2
class C:
    def __init__(self, t3, t4):
        self.t3 = t3
        self.t4 = t4
class P:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


class TestSchemaLinks:

    def test_initialize_link(self):
        """Test basic link usage

        Test uses the following structure:
        class G
            x: class H
              z: 1
            y: Link(x)
        """
        sh = Schema(H, args=[1])
        sg = Schema(G, args=[sh, Link(link_str='x')])
        g = sg.initialize()
        assert g.x is g.y

    def test_initialize_fails_non_existent_link(self):
        """Test link to non-existent referrent fails

        Test uses the following structure:
        class G
            x: Link(k)
            y: class H
              z: 2
        """
        sy = Schema(H, args=[2])
        sg = Schema(G, kwargs={'x':Link(link_str='k'), 'y':sy})
        with pytest.raises(MalformedLinkError):
            g = sg.initialize()

    def test_initialize_fails_non_init_link(self):
        """Test link to non initialized referrent fails

        Test uses the following structure:
        class G
            x: Link(y)
            y: class H
              z: 2
        """
        sy = Schema(H, args=[2])
        sg = Schema(G, kwargs={'x':Link(link_str='y'), 'y':sy})
        with pytest.raises(MalformedLinkError):
            g = sg.initialize()

    def test_initialize_fails_non_init_link_uncle(self):
        """Test link to non initialized referrent fails (diff nesting)

        This differs from previous because Link referrent is at a
        different level of nesting

        Test uses the following structure:
        class G
            x: class H
              z: Link(y)
            y: class H
              z: 2
        """
        sx = Schema(H, args=[Link(link_str='y')])
        sy = Schema(H, args=[2])
        sg = Schema(G, kwargs={'x':sx, 'y':sy})
        with pytest.raises(MalformedLinkError):
            g = sg.initialize()

    def test_initialize_chained_link(self):
        """Test links chained

        Test uses the following structure:
        class P
            a: class A
              t1: class T
                x: class O
                  val: <given>
            b: class B
              t2: Link(a[t1].x)
            c: class C
              t3: Link(a[t1].x)
              t4: Link(b[t2])
        """
        st = Schema(T, args=[Schema(O, kwargs={'val': 1})])
        sa = Schema(A, kwargs={'t1': st})
        sb = Schema(B, kwargs={'t2': Link(link_str='a[t1].x')})
        sc = Schema(C, args=[Link(link_str='a[t1].x'), Link(link_str='b[t2]')])
        sp = Schema(P, kwargs={'a':sa, 'b':sb, 'c': sc})
        p = sp.initialize()
        # All links should be the same object in this case
        assert p.a.t1.x is p.b.t2
        assert p.c.t3 is p.c.t4
        assert p.c.t4 is p.a.t1.x

    def test_initialize_chained_copy_link(self):
        """Test copy links chained

        Test uses the following structure:
        class P
            a: class A
              t1: class T
                x: class O
                  val: <given>
            b: class B
              t2: CopyLink(a[t1].x)
            c: class C
              t3: CopyLink(b[t2])
              t4: Link(b[t2])
        """
        st = Schema(T, args=[Schema(O, kwargs={'val': 1})])
        sa = Schema(A, kwargs={'t1': st})
        sb = Schema(B, kwargs={'t2': CopyLink(link_str='a[t1].x')})
        sc = Schema(C, args=[CopyLink(link_str='b[t2]'), Link(link_str='b[t2]')])
        sp = Schema(P, kwargs={'a':sa, 'b':sb, 'c': sc})
        p = sp.initialize()
        # All links referrents should be different values because copy
        # links are used
        assert p.a.t1.x is not p.b.t2
        assert p.c.t3 is not p.c.t4
        assert p.c.t4 is not p.a.t1.x
        assert p.c.t3 is not p.b.t2
        # Except c.t4 and b.t2 should be the same because normal link
        # was used
        assert p.c.t4 is p.b.t2
        # Still the values should be the same
        assert p.a.t1.x.val == p.b.t2.val
        assert p.c.t3.val == p.c.t4.val
        assert p.c.t4.val == p.a.t1.x.val
        # An update should only affect the object changed
        p.b.t2.val = 2
        assert p.a.t1.x.val != p.b.t2.val
        assert p.c.t4.val == p.b.t2.val
        assert p.c.t3.val == p.a.t1.x.val
