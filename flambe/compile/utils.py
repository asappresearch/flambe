from typing import Type, Set, Any, Optional

from urllib.parse import urlparse


def all_subclasses(class_: Type[Any]) -> Set[Type[Any]]:
    """Return a set of all subclasses for a given class object

    Recursively collects all subclasses of `class_` down the object
    hierarchy into one set.

    Parameters
    ----------
    class_ : Type[Any]
        Class to retrieve all subclasses for

    Returns
    -------
    Set[Type[Any]]
        All subclasses of class_

    """
    subsubclasses = set([s for c in class_.__subclasses__() for s in all_subclasses(c)])
    return set(class_.__subclasses__()).union(subsubclasses)


def make_component(class_: type, tag_namespace: str, only_module: Optional[str] = None) -> None:
    """Make class and all its children a `Component`

    For example a call to `make_component(torch.optim.Adam, "torch")`
    will make the tag `!torch.Adam` accessible in any yaml configs.
    This does *NOT* monkey patch (aka swizzle) torch, but instead
    creates a dynamic subclass which will be used by the yaml
    constructor i.e. only classes loaded from the config will be
    affected, anything imported and used in code.

    Parameters
    ----------
    class_ : type
        To be registered with yaml as a component, along with all its
        children
    tag_prefix : str
        Added to beginning of all the tags
    only_module : str
        Module prefix used to limit the scope of what gets registered

    Returns
    -------
    None

    """
    from flambe.compile import dynamic_component
    for subclass in all_subclasses(class_):
        if only_module is None or subclass.__module__.startswith(only_module):
            dynamic_component(subclass, tag=subclass.__name__, tag_namespace=tag_namespace)


def _is_url(resource: str) -> bool:
    """Whether a given resource is a remote URL.

    Resolve by searching for a scheme.

    Parameters
    ----------
    resource: str
        The given resource

    Returns
    -------
    bool
        If the resource is a remote URL.

    """
    scheme = urlparse(resource).scheme
    return scheme != ''
