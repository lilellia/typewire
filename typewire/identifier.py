from collections.abc import Iterable, Mapping
import types
from typing import Annotated, Any, get_args, get_origin, TypeAlias, Union

TypeHint: TypeAlias = Any


def is_union(type_hint: TypeHint) -> bool:
    """Determine whether the given type represents a union type."""
    if get_origin(type_hint) is Union:
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
    return isinstance(real_type, type) and issubclass(real_type, Mapping)


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

        origin = get_origin(t)
        args = get_args(t)

        if origin is Annotated:
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
