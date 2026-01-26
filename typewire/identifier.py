from collections.abc import Callable, Iterable, Mapping
import sys
import types
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints, is_typeddict, ParamSpec, TypeAlias, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required

if sys.version_info >= (3, 12):
    from typing import TypeAliasType

from .typed_dict import get_typed_dict_key_sets

TypeHint: TypeAlias = Any


def is_union(type_hint: TypeHint) -> bool:
    """Determine whether the given type represents a union type."""
    if get_origin(type_hint) is cast(Any, Union):
        # Union[T1, T2] or Optional[T]
        return True

    if hasattr(types, "UnionType") and isinstance(type_hint, types.UnionType):
        # T1 | T2
        return True

    return type_hint is Union


def is_mapping(type_hint: TypeHint) -> bool:
    """Determine whether the given type represents a mapping type."""
    origin = get_origin(type_hint)
    real_type = origin if origin is not None else type_hint

    # normal mapping classes
    if isinstance(real_type, type) and issubclass(real_type, Mapping):
        return True

    # TypedDict
    if is_typeddict(type_hint):
        return True

    return False


def is_iterable(type_hint: TypeHint) -> bool:
    """Determine whether the given type represents an iterable type."""
    origin = get_origin(type_hint)
    real_type = origin if origin is not None else type_hint
    return isinstance(real_type, type) and issubclass(real_type, Iterable) and real_type not in (str, bytes)


def unwrap(type_hint: TypeHint) -> list[TypeHint]:
    """Recursively unwrap the given type hint, removing Annotated and Union layers."""

    def _flatten(t: TypeHint) -> list[TypeHint]:
        if hasattr(t, "__supertype__"):
            # hint is a NewType
            return _flatten(t.__supertype__)

        if sys.version_info >= (3, 12) and isinstance(t, TypeAliasType):
            # hint is `type Alias = ...` (3.12+ syntax)
            return _flatten(t.__value__)

        origin = get_origin(t)
        args = get_args(t)

        if origin in (cast(Any, T) for T in (Annotated, Required, NotRequired)):
            # hint is Annotated[T, metadata]
            return _flatten(args[0])

        if is_union(t):
            return sum(map(_flatten, args), [])

        return [type(None)] if t is None else [t]

    # return deduplicated list whilst maintaining order
    flattened = _flatten(type_hint)
    unique: list[TypeHint] = []

    for t in flattened:
        if t not in unique:
            unique.append(t)

    return unique


def as_string(type_hint: TypeHint) -> str:
    """Construct a string representation of the given type hint."""
    origin: Any = get_origin(type_hint)
    args: tuple[Any, ...] = get_args(type_hint)

    if type_hint is type(None):
        return "None"

    if type_hint is Ellipsis:
        return "..."

    if is_union(type_hint):
        return " | ".join(map(as_string, args))

    if is_typeddict(type_hint):
        required, optional = get_typed_dict_key_sets(type_hint)
        annot = get_type_hints(type_hint)
        kv_strs: list[str] = []

        for req in required:
            kv_strs.append(f"{req}: Required[{as_string(annot[req])}]")

        for opt in optional:
            kv_strs.append(f"{opt}: NotRequired[{as_string(annot[opt])}]")

        kv_strs.sort()
        return f"{type_hint.__name__}[{', '.join(kv_strs)}]"

    if origin is Callable:
        if not args:
            return "Callable"

        params, return_type = args

        if params is Ellipsis:
            return f"Callable[..., {as_string(return_type)}]"
        elif isinstance(params, ParamSpec):
            return f"Callable[{as_string(params)}, {as_string(return_type)}]"
        else:
            param_str = ", ".join(map(as_string, params))
            return f"Callable[[{param_str}], {as_string(return_type)}]"

    if origin is Annotated:
        T = as_string(args[0]) if args else "Any"
        return f"Annotated[{T}, ...]"

    if origin is not None:
        return f"{as_string(origin)}[{', '.join(map(as_string, args))}]"

    return getattr(type_hint, "__name__", repr(type_hint))
