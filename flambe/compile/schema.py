from __future__ import annotations
import inspect
import itertools
from reprlib import recursive_repr
from collections import ChainMap
from typing import MutableMapping, Any, Callable, Optional, Dict, Sequence
from typing import Tuple, List, Iterator, Union

import copy

from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
                                  CommentedKeySeq, CommentedSeq, TaggedScalar,
                                  CommentedKeyMap)

from flambe.compile.yaml import Registrable, YAMLLoadType


YAML_TYPES = (CommentedMap, CommentedOrderedMap, CommentedSet, CommentedKeySeq, CommentedSeq,
              TaggedScalar, CommentedKeyMap)

KeyType = Union[str, int]
PathType = Tuple[KeyType, ...]


class LinkError(Exception):
    pass


class MalformedLinkError(LinkError):
    pass


class UnpreparedLinkError(LinkError):
    pass


class NonexistentResourceError(LinkError):
    pass


class Options(Registrable):
    pass


def _convert_ints(schematic_path: List[str]) -> List[KeyType]:
    modified: List[KeyType] = list(schematic_path)
    for i, e in enumerate(schematic_path):
        stay_quote = False
        if e.startswith("'") and e.endswith("'") and len(e) > 2:
            e = e[1:-1]
            stay_quote = True
        try:
            integer = int(e)
            if stay_quote:
                # save as string, but leave the quotes stripped
                modified[i] = str(integer)
            else:
                # save as integer
                modified[i] = integer
        except ValueError:
            # The value is not an integer and can be left alone
            pass
    return modified


def _represent_ints(schematic_path: Sequence[KeyType]) -> List[str]:
    modified: List[str] = []
    for e in schematic_path:
        if isinstance(e, int):
            modified.append(str(e))
        else:
            try:
                int(e)
                modified.append("'" + e + "'")
            except ValueError:
                modified.append(e)
    return modified


def create_link_str(schematic_path: Sequence[KeyType],
                    attr_path: Optional[Sequence[str]] = None) -> str:
    """Create a string representation of the specified link

    Performs the reverse operation of
    :func:`~flambe.compile.component.parse_link_str`

    Parameters
    ----------
    schematic_path : Sequence[str]
        List of entries corresponding to dictionary keys in a nested
        :class:`~flambe.compile.Schema`
    attr_path : Optional[Sequence[str]]
        List of attributes to access on the target object
        (the default is None).

    Returns
    -------
    str
        The string representation of the schematic + attribute paths

    Raises
    -------
    MalformedLinkError
        If the schematic_path is empty

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.
    >>> create_link_str(['obj', 'key1', 'key2'], ['attr1', 'attr2'])
    'obj[key1][key2].attr1.attr2'

    """
    if len(schematic_path) == 0:
        raise MalformedLinkError("Can't create link without schematic path")
    schematic_path = _represent_ints(schematic_path)
    root, schematic_path = schematic_path[0], schematic_path[1:]
    schematic_str = ''
    attr_str = ''
    if len(schematic_path) > 0:
        schematic_str = '[' + "][".join(schematic_path) + ']'
    if attr_path is not None and len(attr_path) > 0:
        attr_str = '.' + '.'.join(attr_path)
    return root + schematic_str + attr_str


def parse_link_str(link_str: str) -> Tuple[Tuple[KeyType, ...], Tuple[str, ...]]:
    """Parse link to extract schematic and attribute paths

    Links should be of the format ``obj[key1][key2].attr1.attr2`` where
    obj is the entry point; in a pipeline, obj would be the stage name,
    in a single-object config obj would be the target keyword at the
    top level. The following keys surrounded in brackets traverse
    the nested dictionary structure that appears in the config; this
    is intentonally analagous to how you would access properties in the
    dictionary when loaded into python. Then, you can use the dot
    notation to access the runtime instance attributes of the object
    at that location.

    Parameters
    ----------
    link_str : str
        Link to earlier object in the config of the format
        ``obj[key1][key2].attr1.attr2``

    Returns
    -------
    Tuple[Sequence[str], Sequence[str]]
        Tuple of the schematic and attribute paths respectively

    Raises
    -------
    MalformedLinkError
        If the link is written incorrectly

    Examples
    -------
    Examples should be written in doctest format, and
    should illustrate how to use the function/class.

    >>> parse_link_str('obj[key1][key2].attr1.attr2')
    (('obj', 'key1', 'key2'), ('attr1', 'attr2'))

    """
    schematic_path: List[str] = []
    attr_path: List[str] = []
    temp: List[str] = []
    x = link_str
    # Parse schematic path
    bracket_open = False
    root_extracted = False
    while '[' in x or ']' in x:
        if bracket_open:
            temp = x.split(']', 1)
            if '[' in temp[0]:
                raise MalformedLinkError(f"Previous bracket unclosed in {link_str}")
            if len(temp) != 2:
                # Error case: [ not closed
                raise MalformedLinkError(f"Open bracket '[' not closed in {link_str}")
            schematic_path.append(temp[0])
            bracket_open = False
        else:
            # No bracket open yet
            temp = x.split('[', 1)
            if ']' in temp[0]:
                raise MalformedLinkError(f"Close ']' before open in {link_str}")
            if len(temp) != 2:
                # Error case: ] encountered without [
                raise MalformedLinkError(f"']' encountered before '[' in {link_str}")
            if len(temp[0]) != 0:
                if len(schematic_path) != 0:
                    # Error case: ]text[
                    raise MalformedLinkError(f"Text between brackets in {link_str}")
                # Beginning object name
                schematic_path.append(temp[0])
                root_extracted = True
            else:
                if len(schematic_path) == 0:
                    raise MalformedLinkError(f"No top level object in {link_str}")
            bracket_open = True
        # First part already added to schematic path, keep remainder
        x = temp[1]
    # Parse attribute path
    attr_path = x.split('.')
    if not root_extracted:
        if len(attr_path[0]) == 0:
            raise MalformedLinkError(f"No top level object in {link_str}")
        schematic_path.append(attr_path[0])
    elif len(attr_path) > 1:
        # Schematic processing did happen, so leading dot
        if attr_path[0] != '':
            # Error case: attr without dot beforehand
            raise MalformedLinkError(f"Attribute without preceeding dot notation in {link_str}")
        if attr_path[-1] == '':
            # Error case: trailing dot
            raise MalformedLinkError(f"Trailing dot in {link_str}")
    attr_path = attr_path[1:]
    schematic_path = _convert_ints(schematic_path)
    return tuple(schematic_path), tuple(attr_path)


class Link(Registrable, tag_override="ref"):

    def __init__(self,
                 link_str: Optional[str] = None,
                 schematic_path: Optional[Sequence[KeyType]] = None,
                 attr_path: Optional[Sequence[str]] = None) -> None:
        if link_str is not None:
            if schematic_path is not None or attr_path is not None:
                raise ValueError('If link string is given, no other arguments should be')
            schematic_path, attr_path = parse_link_str(link_str)
        if schematic_path is None:
            raise ValueError('If link string is not given, schematic path is required')
        self.schematic_path = tuple(schematic_path)
        self.attr_path = tuple(attr_path) if attr_path is not None else None
        self.post_init_hooks: List[Callable] = []  # TODO remove pending refactor

    def resolve(self, cache: Dict[PathType, Any]) -> Any:
        try:
            obj = cache[self.schematic_path]
        except KeyError:
            raise UnpreparedLinkError(
                f'Link for schema at {self.schematic_path} does not point to \
                  an object that has been initialized. \
                  Make sure the link points to a non-parent above the link \
                  in the config.'
            )
        if self.attr_path is not None:
            for attr in self.attr_path:
                try:
                    obj = getattr(obj, attr)
                except AttributeError:
                    raise UnpreparedLinkError(f'Link {self} failed. {obj} has no attribute {attr}')
        return obj

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.KWARGS_OR_ARG

    def __call__(self, cache: Dict[PathType, Any]) -> Any:
        return self.resolve(cache)

    def __repr__(self) -> str:
        return f'link({create_link_str(self.schematic_path, self.attr_path)})'


class FileLink(Link, tag_override="file"):

    def __init__(self, file_reference: str):
        self.schematic_path = (file_reference,)
        self.attr_path = None
        self.post_init_hooks: List[Callable] = []  # TODO remove pending refactor

    def resolve(self, cache: Dict[PathType, Any]) -> Any:
        try:
            obj = cache[self.schematic_path]
        except KeyError:
            raise NonexistentResourceError(f'{self.schematic_path[0]} is not available ')
        return obj

    def __repr__(self) -> str:
            return f'file({self.schematic_path[0]})'


class CopyLink(Link, tag_override='copy'):

    def resolve(self, cache: Dict[PathType, Any]) -> Any:
        obj = super().resolve(cache)
        return copy.deepcopy(obj)


VARIABLE_ARG_TYPES = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


class IndexableBoundArguments:

    def __init__(self, ba):
        self._ba = ba
        self.arguments = ba.arguments
        self.signature = ba.signature

    @property
    def args(self):
        return self._ba.args

    @property
    def kwargs(self):
        return self._ba.kwargs

    def _get_varkwargs(self):
        params = self.signature.parameters
        kwargs = [(k, v) for k, v in self.arguments.items() if params[k].kind == inspect.Parameter.VAR_KEYWORD]
        if len(kwargs) > 0:
            assert len(kwargs) == 1
            return kwargs[0]  # tuple with name of kwargs and the dict
        return None, None

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(itertools.chain(self.args, self.kwargs.values()))[key]
        else:
            result = ChainMap(self.kwargs, self.arguments)[key]
            if result not in self.args and result not in self.kwargs.values():
                # Then result is actually a variable arguments group
                raise KeyError(f'Key {key} not in {self}')
            return result

    def _find_by_key(self, key):
        params = self.signature.parameters
        if isinstance(key, int):
            arg_num = 0
            for k, v in self.arguments.items():
                if params[k].kind == inspect.Parameter.VAR_POSITIONAL:
                    idx = key - arg_num
                    if idx >= len(v):
                        arg_num += len(v)
                        continue
                    else:
                        return (self.arguments, k), idx
                elif params[k].kind == inspect.Parameter.VAR_KEYWORD:
                    keys = list(v.keys())
                    idx = key - arg_num
                    if idx >= len(keys):
                        arg_num += len(v)
                        continue
                    else:
                        return v, keys[idx]
                else:
                    if key == arg_num:
                        return self.arguments, k
                    arg_num += 1
        elif isinstance(key, str):
            kwargs_name, kwargs = self._get_varkwargs()
            if key in self.arguments and params[key].kind not in VARIABLE_ARG_TYPES:
                return self.arguments, key
            elif kwargs is not None:
                return self.arguments[kwargs_name], key
        else:
            raise TypeError(f'Key type {type(key)} not supported')
        raise KeyError(f'{key}')

    def __contains__(self, key):
        try:
            _find_by_key(key)
            return True
        except KeyError:
            return False

    def __setitem__(self, key, new_value):
        container, idx = self._find_by_key(key)
        if isinstance(container, tuple):
            a, b = container[0], list(container[0][container[1]])
            b[idx] = new_value
            a[container[1]] = tuple(b)
        else:
            container[idx] = new_value

    def __delitem__(self, key):
        container, idx = self._find_by_key(key)
        if isinstance(container, tuple):
            a, b = container[0], list(container[0][container[1]])
            del b[idx]
            a[container[1]] = tuple(b)
        else:
            del container[idx]

    def _iter_items(self):
        params = self.signature.parameters
        arg_num = 0
        for k, v in self.arguments.items():
            if params[k].kind == inspect.Parameter.VAR_POSITIONAL:
                for e in v:
                    yield arg_num, None, e
                    arg_num += 1
            elif params[k].kind == inspect.Parameter.VAR_KEYWORD:
                for nk, nv in v.items():
                    yield arg_num, nk, nv
                    arg_num += 1
            else:
                yield arg_num, k, v
                arg_num += 1

    def __iter__(self):
        for arg_num, name, value in self._iter_items():
            if name is None:
                yield arg_num
            else:
                yield name

    def __len__(self):
        return len(list(self))

    def named_items(self):
        params = self.signature.parameters
        var_pos = [k for k in self.arguments if params[k].kind == inspect.Parameter.VAR_POSITIONAL]
        if len(var_pos) > 0 and len(self.arguments[var_pos[0]]) > 0:
            # There are at least some variable positional arguments so use indexing by number for
            #  ALL positional args
            yield from enumerate(self.args)
            yield from self.kwargs.items()
        else:
            for arg_num, name, value in self._iter_items():
                if name is None:
                    yield arg_num, value
                else:
                    yield name, value


class Schema(MutableMapping[str, Any]):

    def __init__(self,
                 callable_: Callable,
                 args: Optional[Sequence[Any]] = None,
                 kwargs: Optional[Dict[str, Any]] = None,
                 factory_name: Optional[str] = None,
                 tag: Optional[str] = None,
                 apply_defaults: bool = True,
                 allow_new_args: bool = False):
        self.callable_ = callable_
        self.factory_name = factory_name
        self.apply_defaults = apply_defaults
        if factory_name is None:
            self.factory_method = callable_
        else:
            if not isinstance(self.callable_, type):
                raise ValueError(f'Cannot specify factory name on non-class callable {callable_}')
            self.factory_method = getattr(self.callable_, factory_name)
        args = args if args is not None else []
        kwargs = kwargs if kwargs is not None else {}
        s = inspect.signature(self.factory_method)
        bound_arguments = s.bind(*args, **kwargs)
        if self.apply_defaults:
            bound_arguments.apply_defaults()
        self.bound_arguments = IndexableBoundArguments(bound_arguments)
        if tag is None:
            if isinstance(self.callable_, type):
                tag = type(self.callable_).__name__
            else:
                tag = self.callable_.__name__
        self.created_with_tag = tag
        self.allow_new_args = allow_new_args
        self.post_init_hooks: List[Callable] = []  # TODO remove pending refactor

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType:
        return YAMLLoadType.SCHEMATIC

    @property
    def arguments(self):
        return {k: v for k, v in self.bound_arguments.named_items()}

    def __setitem__(self, key: KeyType, value: Any) -> None:
        self.bound_arguments[key] = value

    def __getitem__(self, key: KeyType) -> Any:
        return self.bound_arguments[key]

    def __delitem__(self, key: KeyType) -> None:
        del self.bound_arguments[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.bound_arguments

    def __len__(self) -> int:
        return len(self.bound_arguments)

    def __call__(self,
                 path: Optional[PathType] = None,
                 cache: Optional[Dict[PathType, Any]] = None,
                 root: Optional[Schema] = None):
        return self.initialize(path, cache, root)

    def __deepcopy__(self, memo=None) -> Schema:
        """Override deepcopy."""
        args = []
        for arg in self.bound_arguments.args:
            if inspect.ismodule(arg) or inspect.ismethod(arg):
                args.append(arg)
            else:
                arg = copy.deepcopy(arg)  # type: ignore
                args.append(arg)

        kwargs = dict()
        for name, arg in self.bound_arguments.kwargs.items():
            if inspect.ismodule(arg) or inspect.ismethod(arg):
                kwargs[name] = arg
            else:
                arg = copy.deepcopy(arg)  # type: ignore
                kwargs[name] = arg

        return Schema(
            callable_=copy.copy(self.callable_),
            args=args,
            kwargs=kwargs,
            tag=self.created_with_tag,
            factory_name=self.factory_name,
            apply_defaults=self.apply_defaults,
            allow_new_args=self.allow_new_args
        )

    @staticmethod
    def traverse(obj: Any,
                 current_path: Optional[PathType] = None,
                 yield_schema: Optional[str] = None) -> Iterator[Tuple[PathType, Any]]:
        current_path = current_path if current_path is not None else tuple()
        if isinstance(obj, Link):
            yield (current_path, obj)
        elif isinstance(obj, Schema):
            if yield_schema is None or yield_schema == 'before':
                yield (current_path, obj)
                yield from Schema.traverse(obj.bound_arguments, current_path, yield_schema)
            elif yield_schema == 'only':
                yield (current_path, obj)
            elif yield_schema == 'after':
                yield from Schema.traverse(obj.bound_arguments, current_path, yield_schema)
                yield (current_path, obj)
            elif yield_schema == 'never':
                yield from Schema.traverse(obj.bound_arguments, current_path, yield_schema)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                next_path = current_path + (k,)
                yield from Schema.traverse(v, next_path, yield_schema)
        elif isinstance(obj, (list, tuple)):
            for i, e in enumerate(obj):
                next_path = current_path[:] + (i,)
                yield from Schema.traverse(e, next_path, yield_schema)
        elif isinstance(obj, IndexableBoundArguments):
            for k, v in obj.named_items():
                next_path = current_path + (k,)
                yield from Schema.traverse(v, next_path, yield_schema)
        else:
            yield (current_path, obj)

    def set_param(self, path: PathType, value: Any):
        """Set path in schema to value

        Convenience method for setting a value deep in a schema. For
        example `root.set_param(('a', 'b', 'c'), val)` is the
        equivalent of `root['a']['b']['c'] = val`. NOTE: you can only
        use set_param on existing paths in the schema. If `c` does not
        already exist in the above example, a `KeyError` is raised.

        Parameters
        ----------
        path : Optional[Tuple[Union[str, int], ...]]
            Description of parameter `path`.
        value : Any
            Description of parameter `value`.

        Raises
        -------
        KeyError
            If any value in the path does not exist as the name of a
            child schema

        """
        current_obj = self
        # ignore type because it's hard to specify that sometimes you
        # can have an int in the path but only if that position is not
        # a schema and actually a list
        try:
            for item in path[:-1]:
                if isinstance(current_obj, (list, tuple)) and not isinstance(item, int):
                    raise KeyError()
                current_obj = current_obj[item]

            last_item = path[-1]
            if isinstance(current_obj, (list, tuple)) and not isinstance(last_item, int):
                raise KeyError()
            current_obj[last_item] = value

        except KeyError:
            raise KeyError(f'{self} has no path {path}. Failed at {last_item}')

    def get_param(self, path: PathType) -> None:
        current_obj = self
        last_item = None
        # ignore type because it's hard to specify that sometimes you
        # can have an int in the path but only if that position is not
        # a schema and actually a list
        try:
            for item in path[:-1]:
                last_item = item
                current_obj = current_obj[item]
            last_item = path[-1]
            return current_obj[last_item]
        except KeyError:
            raise KeyError(f'{self} has no path {path}. Failed at {last_item}')

    def initialize(self,
                   path: Optional[PathType] = None,
                   cache: Optional[Dict[PathType, Any]] = None,
                   root: Optional[Schema] = None) -> Any:
        # Set defaults for values that will be used in recursion
        cache = cache if cache is not None else {}
        path = path if path is not None else tuple()
        new_root = root is None
        # Need to keep a reference to the first caller
        # for set_param updates later
        root = root if root is not None else copy.deepcopy(self)
        initialized = root if new_root else self
        # If object has already been initialized, return that value
        if path in cache:
            return cache[path]

        # TODO remove when possible
        #  legacy / hack for trainer to move things to GPU before
        if hasattr(initialized.factory_method, 'preinitialize'):
            initialized.factory_method.preinitialize(initialized.arguments)

        # Else, recursively initialize all children (bound arguments)
        for current_path, obj in self.traverse(
            initialized.bound_arguments,
            current_path=path,
            yield_schema='only'
        ):
            if isinstance(obj, Link):
                new_value = obj(cache=cache)
                for hook in obj.post_init_hooks:
                    hook(new_value)
                cache[current_path] = new_value
                root.set_param(current_path, new_value)
            elif isinstance(obj, Schema):
                new_value = obj(path=current_path, cache=cache, root=root)
                cache[current_path] = new_value
                root.set_param(current_path, new_value)
            else:
                cache[current_path] = obj
        initialized_arguments = initialized.bound_arguments
        try:
            cache[path] = initialized.factory_method(*initialized_arguments.args,
                                                     **initialized_arguments.kwargs)
            for hook in initialized.post_init_hooks:
                hook(cache[path])
        except TypeError as te:
            print(f"Constructor {self.factory_method} failed with "
                  f"arguments:\n{initialized_arguments.arguments.items()}")
            raise te
        return cache[path]

    def extract_search_space(self) -> Dict[PathType, Options]:
        search_space = {}

        for path, item in Schema.traverse(self, yield_schema='never'):
            if isinstance(item, Link):
                pass
            elif isinstance(item, Options):
                search_space[path] = item

        return search_space

    def extract_links(self, include_files: bool = False) -> List[Link]:
        links = []

        for path, item in Schema.traverse(self, yield_schema='never'):
            if isinstance(item, Link):
                if not isinstance(item, FileLink) or include_files:
                    links.append(item)

        return links

    def set_from_search_space(self, search_space: Dict[PathType, Any]):
        for path, value in search_space.items():
            self.set_param(path, value)

    @recursive_repr()
    def __repr__(self) -> str:
        args = ", ".join("{}={!r}".format(k, v) for k, v in self.arguments.items())
        format_string = "{module}.{cls}({callable_}, {args})"
        return format_string.format(module=self.__class__.__module__,
                                    cls=self.__class__.__qualname__,
                                    tag=self.created_with_tag,
                                    callable_=self.callable_,
                                    factory_method=self.factory_method,
                                    args=args)
