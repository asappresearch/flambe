from typing import Callable, Optional, Any, Union, TextIO, Dict, Mapping, Type, Sequence, Iterable
from typing_extensions import Protocol, runtime_checkable
from enum import Enum
import functools
from warnings import warn
import importlib
import inspect
import io

import ruamel.yaml


TAG_DELIMETER = '.'
TAG_BEGIN = '!'


class MalformedConfig(Exception):
    pass


class YAMLLoadType(Enum):
    """Defines the way a class should be loaded and dumped in YAML

    SCHEMATIC indicates that a Schema should be created using the given
    arguments which can be positional or keyword args.

    KWARGS indicates the arguments will always be a dicitonary mapping
    keywords to the arguments and that dict will be flatted when passed
    to the given callable via **

    KWARGS_OR_ARG is the same as KWARGS except if the given argument is
    not a dictionary, as a fallback it will be passed directly to the
    given callable

    KWARGS_OR_POSARGS is the same as KWARGS except if the given
    argument is not a dictionary BUT IS a list, then that argument
    will be expanded via *

    A Class can choose how to be treated by implementing the
    TypedYAMLLoad protocol below.

    """
    SCHEMATIC = 'schematic'  # all delayed init objs
    KWARGS = 'kwargs'  # expands raw_obj via **
    KWARGS_OR_ARG = 'kwargs_or_arg'  # expands via ** if raw_obj is dict, else don't expand
    KWARGS_OR_POSARGS = 'kwargs_or_posargs' # expands via ** if raw_obj is dict, else expands via *


@runtime_checkable
class TypedYAMLLoad(Protocol):
    """Used to tell flambe what YAMLLoadType the class is"""

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType: ...


@runtime_checkable
class CustomYAMLLoad(Protocol):
    """Used to implement custom YAML representations"""

    @classmethod
    def from_yaml(cls, raw_obj: Any, callable_override: Optional[Callable] = None) -> Any: ...
    """Produce new object using raw object ruamel.yaml loaded"""

    @classmethod
    def to_yaml(cls, instance: Any) -> Any: ...
    """Produce ruamel.yaml compatible object from instance"""


class Registrable:
    """Subclasses automatically registers for top level tag usage

    Subclasses of this class can be used without any import path in the
    YAML tag. For example, instead of `!flambe.nn.RNNEncoder` you can
    use `!RNNEncoder` because this class registers class names which
    are eventually used by the load_config function as a top level
    lookup table
    """
    tag_to_class = {}

    def __init_subclass__(cls: Type['Registrable'],
                          tag_override: Optional[str] = None,
                          **kwargs: Mapping[str, Any]) -> None:
        tag = tag_override if tag_override is not None else cls.__qualname__
        if tag in Registrable.tag_to_class:
            assert Registrable.tag_to_class[tag].__name__ == cls.__name__
            warn(f"Re-registration of tag with class = {cls}")
        Registrable.tag_to_class[tag] = cls


class _DefaultYAMLLoader:
    """Stub class used when binding from_yaml and to_yaml methods"""
    pass


def _is_tagged_null(obj):
    """Detects if Scalar object corresponds to a null value

    NOTE: that it's equivalent to an empty string; but in the context
    of where this function is used, an empty string should correspond
    to Null

    TODO investigate robust check here
    """
    return isinstance(obj, ruamel.yaml.comments.TaggedScalar) and obj.value == ''


def _resolve_callable(cls, callable_override):
    """Given cls and callable determines which should be used"""
    if callable_override is None:
        if cls is _DefaultYAMLLoader:
            raise Exception("No class or callable specified; Fatal error")
        if inspect.isabstract(cls):
            msg = f"You're trying to initialize an abstract class {cls.__name__}. " \
                  + "If you think it's concrete, double check you've spelled " \
                  + "all the originally abstract method names correctly."
            raise Exception(msg)
        return cls
    else:
        return callable_override


def _pass_kwargs(callable_, kwargs):
    if kwargs is None:
        raise Exception('None for kwargs for {callable_}')
    try:
        instance = callable_(**kwargs)
    except TypeError as te:
        raise TypeError(f"Initializing {callable_} failed with keyword args {kwargs}") from te
    instance._saved_arguments = kwargs
    return instance


def _pass_arg(callable_, arg):
    if arg is None:
        raise Exception('None for arg for {callable_}; should be null Tagged Scalar')

    processed = saved_arg = arg
    if isinstance(arg, ruamel.yaml.comments.TaggedScalar):
        processed = arg.value if arg.value != '' else None
        saved_arg = ruamel.yaml.comments.CommentedMap()
        arg.copy_attributes(saved_arg)
    try:
        instance = callable_(processed) if processed is not None else callable_()
    except TypeError as te:
        raise TypeError(f"Initializing {callable_} failed with argument {arg}") from te
    instance._saved_arguments = saved_arg
    return instance


def _pass_posargs(callable_, posargs):
    if posargs is None:
        raise Exception('None for positional args for {callable_}')
    try:
        instance = callable_(*posargs)
    except TypeError as te:
        raise TypeError(f"Initializing {callable_} failed with positional args {posargs}") from te
    instance._saved_arguments = posargs
    return instance


def schematic_from_yaml(cls, raw_obj: Any, callable_override: Optional[Callable] = None) -> 'Schema':
    from flambe.compile.schema import Schema
    tag = raw_obj.tag.value
    if isinstance(raw_obj, dict):
        return Schema(callable_override, kwargs=raw_obj, tag=tag)
    if isinstance(raw_obj, list):
        return Schema(callable_override, args=raw_obj, tag=tag)
    if _is_tagged_null(raw_obj):
        return Schema(callable_override, tag=tag)
    if raw_obj is not None:
        return Schema(callable_override, args=[raw_obj], tag=tag)
    return Schema(callable_override, tag=tag)


def schematic_to_yaml(cls, schema: 'Schema') -> Any:
    kwargs = schema.arguments
    y = ruamel.yaml.comments.CommentedMap(kwargs)
    y.yaml_set_tag(schema.created_with_tag)
    return y


def kwargs_from_yaml(cls, raw_obj: Any, callable_override: Optional[Callable] = None) -> Any:
    callable_ = _resolve_callable(cls, callable_override)
    if isinstance(raw_obj, dict):
        return _pass_kwargs(callable_, raw_obj)
    elif _is_tagged_null(raw_obj):
        return _pass_arg(callable_, raw_obj)
    raise Exception(f'Unsupported argument type {type(raw_obj)} for {callable_}')


def kwargs_or_arg_from_yaml(cls,
                            raw_obj: Any,
                            callable_override: Optional[Callable] = None) -> Any:
    callable_ = _resolve_callable(cls, callable_override)
    if isinstance(raw_obj, dict):
        return _pass_kwargs(callable_, raw_obj)
    return _pass_arg(callable_, raw_obj)


def kwargs_or_posargs_from_yaml(cls,
                                raw_obj: Any,
                                callable_override: Optional[Callable] = None) -> Any:
    callable_ = _resolve_callable(cls, callable_override)
    if isinstance(raw_obj, dict):
        return _pass_kwargs(callable_, raw_obj)
    elif isinstance(raw_obj, (list, tuple)):
        return _pass_posargs(callable_, raw_obj)
    elif _is_tagged_null(raw_obj):
        return _pass_arg(callable_, raw_obj)
    raise Exception(f'Unsupported argument type {type(raw_obj)} for {callable_}')


def genargs_to_yaml(cls, instance: Any) -> Any:
    if hasattr(instance, '_saved_arguments'):
        return instance._saved_arguments
    raise Exception(f'{instance} doesnt have any attribute indicating args for YAML representation')


def load_type_from_yaml(load_type: YAMLLoadType,
                        cls: Optional[Type] = None,
                        callable_: Optional[Callable] = None) -> Callable:
    cls = cls if cls is not None else _DefaultYAMLLoader
    load_type = YAMLLoadType(load_type)
    # Bind methods to cls via __get__
    if load_type == YAMLLoadType.SCHEMATIC:
        return functools.partial(schematic_from_yaml.__get__(cls), callable_override=callable_)
    elif load_type == YAMLLoadType.KWARGS:
        return kwargs_from_yaml.__get__(cls)
    elif load_type == YAMLLoadType.KWARGS_OR_ARG:
        return kwargs_or_arg_from_yaml.__get__(cls)
    elif load_type == YAMLLoadType.KWARGS_OR_POSARGS:
        return kwargs_or_posargs_from_yaml.__get__(cls)
    raise Exception(f'Creating YAML loader failed with args: {{load_type={load_type}, cls={cls}, '
                    f'callable_={callable_}}}')


def load_type_to_yaml(load_type: YAMLLoadType,
                      cls: Optional[Type] = None,
                      callable_: Optional[Callable] = None) -> Callable:
    cls = cls if cls is not None else _DefaultYAMLLoader
    load_type = YAMLLoadType(load_type)
    if load_type == YAMLLoadType.SCHEMATIC:
        return schematic_to_yaml.__get__(cls)
    elif load_type in [YAMLLoadType.KWARGS, YAMLLoadType.KWARGS_OR_ARG, YAMLLoadType.KWARGS_OR_POSARGS]:
        return genargs_to_yaml.__get__(cls)
    else:
        raise Exception('load to yaml todo')


def _combine_tag(*args):
    if len(args) == 0:
        raise ValueError('Must have >= 1 tag elements to combine')
    if args[0] == '':
        args = args[1:]
        if len(args) == 0:
            raise ValueError('Must have >= 1 tag elements to combine; only element was empty')
    if '' in args:
        raise ValueError('No tag elements can be empty strings except the first (root namespace)')
    raw_text = ''.join(args)
    if TAG_BEGIN in raw_text:
        raise ValueError('')
    return TAG_BEGIN + TAG_DELIMETER.join(args)


def _split_tag(tag) -> str:
    if tag[0] != TAG_BEGIN:
        raise ValueError(f'Invalid tag {tag}')
    tag_components = tag.split(TAG_BEGIN)[1].split(TAG_DELIMETER)
    if len(tag_components) == 0:
        raise ValueError(f'Invalid tag {tag}')
    return tag_components


def fetch_callable(path, begin=None):
    if begin is None:
        mod = importlib.import_module(path[0])
        begin = mod
    obj = begin
    for a in path[1:]:
        obj = getattr(obj, a)
    if obj is None:
        raise Exception(f'Tag to {path} references None value')
    return obj


def create(obj: Any, lookup) -> Any:
    if hasattr(obj, '_yaml_tag'):
        original_tag = obj._yaml_tag.value
        tag = _split_tag(original_tag)
        if tag[0] in lookup:
            callable_ = fetch_callable(tag[1:], lookup[tag[0]])
        else:
            callable_ = fetch_callable(tag)
        cls = None
        if hasattr(callable_, '__self__'):
            # callable_ is a classmethod
            cls = callable_.__self__
        elif isinstance(callable_, type):
            # callable is a class
            cls = callable_
        from_yaml = None
        if cls is not None:
            # Check how class wants to be treated
            if isinstance(cls, TypedYAMLLoad):
                proto = TypedYAMLLoad
                load_type = cls.yaml_load_type()
                from_yaml = load_type_from_yaml(load_type, cls, callable_)
            elif isinstance(cls, CustomYAMLLoad):
                raise NotImplementedError(f'{cls} -- Currently no native objects support this so disabling')
                proto = CustomYAMLLoad
                from_yaml = cls.from_yaml
        from_yaml = from_yaml if from_yaml is not None else load_type_from_yaml(YAMLLoadType.SCHEMATIC, cls, callable_)
        if callable_ != cls:
            instance = from_yaml(obj, callable_override=callable_)
        else:
            instance = from_yaml(obj)
        instance._yaml_tag = original_tag
        return instance
    return obj


def convert_tagged_objects(obj: Any, lookup: Dict[str, Any]) -> Any:
    """Converts into objects corresponding to the given import tags"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = convert_tagged_objects(v, lookup)
        obj = create(obj, lookup)
    elif isinstance(obj, (list, tuple)):
        for i, e in enumerate(obj):
            obj[i] = convert_tagged_objects(e, lookup)
        obj = create(obj, lookup)
    else:
        obj = create(obj, lookup)
    return obj


def convert_objects_to_tagged(obj: Any) -> Any:
    """Converts objects to ruamel.yaml compatible tagged objects"""
    if isinstance(obj, TypedYAMLLoad):
        load_type = type(obj).yaml_load_type()
        to_yaml = load_type_to_yaml(load_type, type(obj))
        y = to_yaml(obj)
    else:
        y = obj
    if isinstance(y, dict):
        for k, v in y.items():
            y[k] = convert_objects_to_tagged(v)
    elif isinstance(y, (list, tuple)):
        for i, e in enumerate(y):
            y[i] = convert_objects_to_tagged(e)
    return y


def load_config(yaml_config: Union[TextIO, str]) -> Iterable[Any]:
    """Load yaml config

    This function will read extensions from the YAML if any, update
    the registry, sync the registry with YAML, and then load the
    object from the YAML representation.

    Parameters
    ----------
    yaml_config: Union[TextIO, str]
        flambe config as a string, or file pointer (result from
        file.read())

    Returns
    -------
    Iterable[Any]
        Iterable over the objects defined in the YAML for each YAML
        document in the config. Note that the number of items in the
        iterable corresponds to the number of YAML documents which are
        separated by '---'

    """
    yaml = ruamel.yaml.YAML()
    lookup = Registrable.tag_to_class
    results = list(yaml.load_all(yaml_config))
    for result in results:
        yield convert_tagged_objects(result, lookup)


def load_first_config(yaml_config: Union[TextIO, str]) -> Any:
    """Load first yaml document from the config"""
    return next(load_config(yaml_config))


def load_config_from_file(file_path: str) -> Iterable[Any]:
    """Load config after reading it from the file path

    Parameters
    ----------
    file_path : str
        Location of YAML config file.

    Returns
    -------
    Iterable[Any]
        See load_config

    Raises
    -------
    FileNotFoundError
        If the specified file is not found

    """
    with open(file_path) as f:
        yield from load_config(f.read())


def load_first_config_from_file(file_path: str) -> Any:
    """Load first config after reading it from the file path"""
    with open(file_path) as f:
        return load_first_config(f.read())


def num_yaml_files(file_path: str) -> int:
    """Load first config after reading it from the file path"""
    with open(file_path) as f:
        yaml = ruamel.yaml.YAML()
        results = list(yaml.load_all(f.read()))
    return len(results)


def dump_config(objs: Sequence[Any],
                stream: Optional[Any] = None) -> Optional[str]:
    """Dump the given objects into the stream.

    Only dump objects that inherit from a Flambé class, or that have
    been manually registered in the registry. If stream is None, the
    config will be returned as a string.

    Parameters
    ----------
    obj : Sequence[Any]
        Objects to be serialized into YAML. State will not be included,
        only kwargs used to initialize the object in the same way
    stream : Optional[Any]
        File stream to dump to; will be forwarded to yaml.dump().
        (default is None)

    """
    if len(objs) == 0:
        raise ValueError("At least one object is required.")
    yaml = ruamel.yaml.YAML()
    # Add representer that can capture tags
    results = []
    for obj in objs:
        results.append(convert_objects_to_tagged(obj))
    return_string = False
    if stream is None:
        return_string = True
        stream = io.StringIO()
    if len(results) > 1:
        yaml.dump_all(results, stream)
    else:
        yaml.dump(results[0], stream)
    if return_string:
        config = stream.getvalue()
        stream.close()
        return config


def dump_one_config(obj: Any, stream: Optional[Any] = None) -> Optional[str]:
    """Dump the given object into the stream."""
    return dump_config([obj], stream)
