from collections.abc import Iterable, Mapping
from contextlib import suppress
import inspect
import types
from typing import Annotated, Any, get_args, get_origin, Literal, TypeAlias, TypeVar, Union

TypeHint: TypeAlias = Any


def is_union(type_hint: TypeHint) -> bool:
    """Determine whether the given type represents a union type."""
    return type_hint is Union or (hasattr(types, "UnionType") and isinstance(type_hint, types.UnionType))


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


def as_type(value: Any, to: TypeHint, *, transparent_int: bool = False, semantic_bool: bool = False) -> Any:
    """Cast a value to the given type hint.

    :param value:
        The raw input value to cast.
    :param to:
        The type hint to cast to.
    :param transparent_int:
        Whether to allow more transparent casting to int.
        For example, int("1.0") raises a ValueError, so as_type("1.0", int) raises a ValueError as well.
        However, as_type("1.0", int, transparent_int=True) will return 1.
        This passes the conversion to float, then int, so as_type("1.3", int, transparent_int=True) returns 1.
    :param semantic_bool:
        Whether to allow for more semantic casting to bool.
        For example, bool("false") returns True, so as_type("false", bool) returns True.
        However, as_type("false", bool, semantic_bool=True) returns False.

    :return: The casted value.
    """

    # We can't cast to Any or an unbound TypeVar, so just return the value as-is
    if to is Any or isinstance(to, TypeVar):
        return value

    origin: Any = get_origin(to)
    args: Any = get_args(to)

    # reach into Annotated
    if origin is Annotated:
        to = get_args(to)[0]
        origin = get_origin(to)
        args = get_args(to)

    # handle unions
    if is_union(to):
        for type_hint in get_args(to):
            if type_hint is type(None) and value is None:
                return None

            with suppress(ValueError, TypeError):
                return as_type(value, type_hint, transparent_int=transparent_int, semantic_bool=semantic_bool)
        else:
            raise ValueError(f"Value {value!r} does not match any type in {to}")

    # handle literals
    if origin is Literal:
        if value in args:
            return value

        raise ValueError(f"Value {value!r} does not match any literal in {to}")

    # If `to` is a plain type (e.g., int), then origin is None. But we want something we can actually call.
    real_type = origin if origin is not None else to

    # handle unions (e.g., int | float | None)
    if is_union(real_type):
        for type_hint in args:
            # if value is None, then we can allow None in the union
            if type_hint is type(None):
                if value is None:
                    return None
                continue

            # otherwise, try to find the first matching type
            with suppress(ValueError, TypeError):
                return as_type(value, type_hint, transparent_int=transparent_int, semantic_bool=semantic_bool)
        else:
            # none of the types match
            raise ValueError(f"Value {value!r} does not match any type in {to}")

    # handle mappings
    if is_mapping(real_type):
        if not isinstance(value, Mapping):
            # input is a list of pairs like [("a", 1), ("b", 2)]
            try:
                value = dict(value)
            except ValueError:
                raise ValueError(f"Value {value!r} is not a mapping")

        key_type = args[0] if args else Any
        val_type = args[1] if len(args) > 1 else Any

        dct = {
            as_type(key, key_type, transparent_int=transparent_int, semantic_bool=semantic_bool): as_type(
                val, val_type, transparent_int=transparent_int, semantic_bool=semantic_bool
            )
            for key, val in value.items()
        }

        if inspect.isabstract(real_type) and isinstance(value, real_type):
            # We can't cast to an abstract container, so just return the dict that we have
            return dct

        return real_type(dct)

    # handle containers
    if is_iterable(real_type):
        if isinstance(value, (str, bytes)) and isinstance(value, real_type):
            # specifically handle Iterable[str] and Iterable[bytes] as simply str and bytes
            return value

        # default to str if the inner type is not set, e.g. x: list
        inner_type = args[0] if args else Any

        # if tuple[T, T] fixed length
        if origin is tuple and args and Ellipsis not in args:
            if len(args) != len(value):
                raise ValueError(f"Expected tuple of length {len(args)}, got {len(value)}")

            return tuple(
                as_type(v, t, transparent_int=transparent_int, semantic_bool=semantic_bool) for v, t in zip(value, args)
            )

        # otherwise, it's a variadic container
        vals = (
            as_type(
                v,
                inner_type,
                transparent_int=transparent_int,
                semantic_bool=semantic_bool,
            )
            for v in value
        )

        if inspect.isabstract(real_type):
            # We can't cast to an abstract container, so just return the value as a list
            return list(vals)

        return real_type(vals)

    # handle possible semantic conversions
    if to is int and transparent_int:
        with suppress(ValueError, TypeError):
            return int(float(value))

    if to is bool and semantic_bool and isinstance(value, str):
        normalized = value.lower()

        if normalized in ("true", "yes", "1", "on"):
            return True

        if normalized in ("false", "no", "0", "off"):
            return False

    if isinstance(real_type, type) and callable(real_type):
        if inspect.isabstract(real_type):
            # We can't instantiate an abstract class, so just return the value
            return value

        return real_type(value)

    # fallback
    return to(value)
