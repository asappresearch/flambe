from typing import Callable, Optional, Any, Union, TextIO, Dict, List, Mapping, Type
from typing_extensions import Protocol, runtime_checkable
from enum import Enum
import functools
from warnings import warn
import importlib
import inspect

import ruamel.yaml

from flambe.compile.extensions import import_modules, is_installed_module


TAG_DELIMETER = '.'
TAG_BEGIN = '!'


class MalformedConfig(Exception):
    pass


class YAMLLoadType(Enum):
    SCHEMATIC = 'schematic'  # all delayed init objs
    KWARGS = 'kwargs'  # expands raw_obj via **
    KWARGS_OR_ARG = 'kwargs_or_arg'  # expands via ** if raw_obj is dict, else don't expand
    KWARGS_OR_POSARGS = 'kwargs_or_posargs' # expands via ** if raw_obj is dict, else expands via *


@runtime_checkable
class TypedYAMLLoad(Protocol):

    @classmethod
    def yaml_load_type(cls) -> YAMLLoadType: ...


@runtime_checkable
class CustomYAMLLoad(Protocol):

    @classmethod
    def from_yaml(cls, raw_obj: Any, callable_override: Optional[Callable] = None) -> Any: ...

    @classmethod
    def to_yaml(cls, instance: Any) -> Any: ...


class Registrable:
    """Subclasses automatically registered as yaml tags

    Automatically registers subclasses with the yaml loader by
    adding a constructor and representer which can be overridden
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_created_with_tag'):
            self._created_with_tag = None


class DefaultYAMLLoader:
    pass


def _resolve_callable(cls, callable_override):
    if callable_override is None:
        if cls is DefaultYAMLLoader:
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
    try:
        instance = callable_(**kwargs) if kwargs is not None else callable_()
    except TypeError as te:
        raise TypeError(f"Initializing {callable_} failed with keyword args {kwargs}") from te
    instance._saved_kwargs = kwargs
    return instance

def _pass_arg(callable_, arg):
    try:
        instance = callable_(arg) if arg is not None else callable_()
    except TypeError as te:
        raise TypeError(f"Initializing {callable_} failed with argument {arg}") from te
    instance._saved_arg = arg
    return instance

def _pass_posargs(callable_, posargs):
    try:
        instance = callable_(*posargs) if posargs is not None else callable_()
    except TypeError as te:
        raise TypeError(f"Initializing {callable_} failed with positional args {posargs}") from te
    instance._saved_posargs = posargs
    return instance


def schematic_from_yaml(cls, raw_obj: Any, callable_override: Optional[Callable] = None) -> 'Schema':
    from flambe.compile.schema import Schema
    if isinstance(raw_obj, dict):
        return Schema(callable_override, kwargs=raw_obj)
    if isinstance(raw_obj, list):
        return Schema(callable_override, args=raw_obj)
    if raw_obj is not None:
        return Schema(callable_override, args=[raw_obj])
    return Schema(callable_override)


def kwargs_from_yaml(cls, raw_obj: Any, callable_override: Optional[Callable] = None) -> Any:
    callable_ = _resolve_callable(cls, callable_override)
    if isinstance(raw_obj, dict):
        return _pass_kwargs(callable_, raw_obj)
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
    raise Exception(f'Unsupported argument type {type(raw_obj)} for {callable_}')


def genargs_to_yaml(cls, instance: Any) -> Any:
    if hasattr(instance, '_saved_kwargs'):
        return instance._saved_kwargs
    elif hasattr(instance, '_saved_arg'):
        return instance._saved_arg
    elif hasattr(instance, '_saved_posargs'):
        return instance._saved_posargs
    raise Exception(f'{instance} doesnt have')


def load_type_from_yaml(load_type: YAMLLoadType,
                        cls: Optional[Type] = None,
                        callable_: Optional[Callable] = None) -> Callable:
    cls = cls if cls is not None else DefaultYAMLLoader
    # Bind methods to cls via __get__
    if load_type == YAMLLoadType.SCHEMATIC:
        return functools.partial(schematic_from_yaml.__get__(cls), callable_override=callable_)
    elif load_type == YAMLLoadType.KWARGS:
        return kwargs_from_yaml.__get__(cls)
    elif load_type == YAMLLoadType.KWARGS_OR_ARG:
        return kwargs_or_arg_from_yaml.__get__(cls)
    elif load_type == YAMLLoadType.KWARGS_OR_POSARGS:
        return kwargs_or_posargs_from_yaml.__get__(cls)
    else:
        raise Exception(f'Load type {load_type} for callable {callable_} not supported.')


def load_type_to_yaml(load_type: YAMLLoadType,
                      cls: Optional[Type] = None,
                      callable_: Optional[Callable] = None) -> Callable:
    cls = cls if cls is not None else DefaultYAMLLoader
    if load_type == YAMLLoadType.SCHEMATIC:
        pass
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
        tag = _split_tag(obj._yaml_tag.value)
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
                # to_yaml = load_type_to_yaml(load_type)
            elif isinstance(cls, CustomYAMLLoad):
                # print(cls)
                raise NotImplementedError(f'{cls} -- Currently no native objects support this so disabling')
                proto = CustomYAMLLoad
                from_yaml = cls.from_yaml
                # to_yaml = cls.to_yaml
        from_yaml = from_yaml if from_yaml is not None else load_type_from_yaml(YAMLLoadType.SCHEMATIC, cls, callable_)
        # to_yaml = to_yaml if to_yaml is not None else load_type_to_yaml(YAMLLoadType.SCHEMATIC, cls, callable_)
        if isinstance(obj, ruamel.yaml.comments.TaggedScalar):
            obj = obj.value if obj.value != '' else None
        if callable_ != cls:
            return from_yaml(obj, callable_override=callable_)
        return from_yaml(obj)
    return obj


def convert_tagged_primatives_to_objects(obj: Any, lookup: Dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = convert_tagged_primatives_to_objects(v, lookup)
        obj = create(obj, lookup)
    elif isinstance(obj, (list, tuple)):
        for i, e in enumerate(obj):
            obj[i] = convert_tagged_primatives_to_objects(e, lookup)
        obj = create(obj, lookup)
    else:
        obj = create(obj, lookup)
    return obj


def _load_environment(yaml_config: Any) -> Dict[str, Any]:
    vanilla_yaml = ruamel.yaml.YAML()
    try:
        yamls = list(vanilla_yaml.load_all(yaml_config))
    except TypeError as e:
        # TODO
        raise MalformedConfig('TODO more info on this error type')
    if len(yamls) > 2:
        raise ValueError(f"{len(yamls)} yaml streams found in file {yaml_config}. "
                         "A file should contain an (optional) environment section " +
                         "and the main runnable object (<= 2 streams separated by '---').")
    environment: Dict[str, Any] = {}
    if len(yamls) == 2:
        environment = dict(yamls[0])
    return environment

# TODO should we tighten signatures (below) Any -> schema?


def load_config(yaml_config: Union[TextIO, str]) -> Any:
    """Load yaml config using the flambe registry

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
    Any
        Initialized object defined by the YAML config

    Raises
    -------
    MalformedConfig
        TODO
    ValueError
        If the file does not contain either 1 or 2 YAML documents
        separated by '---'

    """
    extensions = _load_environment(yaml_config)
    for module in extensions.keys():
        if not is_installed_module(module):
            raise ImportError(
                f"Module {x} is required and not installed. Please 'pip install'"
                "the package containing the module or set auto_install flag"
                " to True."
            )
    yaml = ruamel.yaml.YAML()
    lookup = Registrable.tag_to_class
    result = list(yaml.load_all(yaml_config))[-1]
    result = convert_tagged_primatives_to_objects(result, lookup)
    return result


def load_config_from_file(file_path: str) -> Any:
    """Load config after reading it from the file path

    Parameters
    ----------
    file_path : str
        Location of YAML config file.

    Returns
    -------
    Any
        Initialized object defined by the YAML config

    Raises
    -------
    FileNotFoundError
        If the specified file is not found
    MalformedConfig
        TODO
    ValueError
        If the file does not contain either 1 or 2 YAML documents
        separated by '---'

    """
    with open(file_path) as f:
        result = load_config(f.read())
    return result


def dump_config(obj: Any, stream: Any, environment: Optional[Dict[str, Any]] = None):
    """Dump the given object into the stream including the environment.

    Only dump objects that inherit from a Flambé class, or that have
    been manually registered in the registry.

    Parameters
    ----------
    obj : Any
        Object to be serialized into YAML. State will not be included,
        only kwargs used to initialize the object in the same way;
        Object must be serializable, meaning an entry exists in the
        registry.
    stream : Any
        File stream to dump to; will be forwarded to yaml.dump()
    environment : Optional[Dict[str, Any]]
        Envrionment parameters.
        Default is None.

    """
    environment = environment or {}
    if len(environment) > 0:
        yaml.dump_all([environment, obj], stream)
    else:
        yaml.dump(obj, stream)


def load_environment(yaml_config: Union[TextIO, str]) -> Dict[str, Any]:
    """Load environment from a flambe YAML config file

    The environment should be the first document in a two document YAML
    file where the sections are separated by '---'. If no environment
    section is present, an empty dictionary will be returned.

    Parameters
    ----------
    yaml_config : Union[TextIO, str]
        flambe config as a string, or file path pointing to config

    Returns
    -------
    Dict[str, Any]
        A mapping from module names to package names (and versions).
        Default is None.

    Raises
    -------
    FileNotFoundError
        If the specified file is not found
    MalformedConfig
        TODO
    ValueError
        If the file does not contain 1 or 2 YAML documents
        separated by '---'

    """
    extensions = _load_environment(yaml_config)
    return extensions


def load_environment_from_file(file_path: str) -> Dict[str, Any]:
    """Load environment after reading it from the file path.

    Parameters
    ----------
    file_path : str
        Location of YAML config file.

    Returns
    -------
    Dict[str, Any]
        Initialized object defined by the YAML config

    Raises
    -------
    FileNotFoundError
        If the specified file is not found
    MalformedConfig
        TODO
    ValueError
        If the file does not contain either 1 or 2 YAML documents
        separated by '---'

    """
    with open(file_path) as f:
        result = load_environment(f.read())
    return result
