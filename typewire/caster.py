from collections.abc import Mapping
from contextlib import suppress
import inspect
from typing import Annotated, Any, get_args, get_origin, Literal, TypeVar

from .identifier import is_iterable, is_mapping, is_typed_dict, is_union, TypeHint


def as_type(
    value: Any,
    to: TypeHint,
    *,
    transparent_int: bool = False,
    semantic_bool: bool = False,
    closed_typed_dicts: bool = False,
) -> Any:
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
    :param closed_typed_dicts:
        When `to` is (or contains) a TypedDict, this determines whether additional keys beyond the TypedDict's schema
        are allowed. With `closed_typed_dicts=True`, additional keys will raise a `ValueError`. That is,

        >>> class Point(TypedDict):
        ...     x: float
        ...     y: float

        >>> as_type({"x": "1.0", "y": "2.0"}, Point, closed_typed_dicts=False)
        {'x': 1.0, 'y': 2.0}
        >>> as_type({"x": "1.0", "y": "2.0"}, Point, closed_typed_dicts=True)
        ValueError("Unexpected field(s) for Point: 'z'")

    :return: The casted value.
    """
    kwargs = {
        "transparent_int": transparent_int,
        "semantic_bool": semantic_bool,
        "closed_typed_dicts": closed_typed_dicts,
    }

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
        if value is None and type(None) in args:
            # if we're allowed to have None in the union, then return that
            return None

        for type_hint in get_args(to):
            with suppress(ValueError, TypeError):
                return as_type(value, type_hint, **kwargs)
        else:
            raise ValueError(f"Value {value!r} does not match any type in {to}")

    # handle literals
    if origin is Literal:
        if value in args:
            return value

        raise ValueError(f"Value {value!r} does not match any literal in {to}")

    # If `to` is a plain type (e.g., int), then origin is None. But we want something we can actually call.
    real_type = origin if origin is not None else to

    # handle mappings
    if is_mapping(real_type) and not is_typed_dict(real_type):
        if not isinstance(value, Mapping):
            # input is a list of pairs like [("a", 1), ("b", 2)]
            try:
                value = dict(value)
            except ValueError:
                raise ValueError(f"Value {value!r} is not a mapping")

        key_type = args[0] if args else Any
        val_type = args[1] if len(args) > 1 else Any

        dct = {as_type(key, key_type, **kwargs): as_type(val, val_type, **kwargs) for key, val in value.items()}

        if inspect.isabstract(real_type) and isinstance(value, real_type):
            # We can't cast to an abstract container, so just return the dict that we have
            return dct

        return real_type(dct)

    # handle TypedDict
    if is_typed_dict(real_type):
        if not isinstance(value, Mapping):
            # input is a list of pairs like [("a", 1), ("b", 2)]
            try:
                value = dict(value)
            except ValueError:
                raise ValueError(f"Value {value!r} is not a mapping")

        annot = real_type.__annotations__

        # perform casting
        dct = {key: as_type(val, annot.get(key, Any), **kwargs) for key, val in value.items()}

        # perform validation
        keys = set(dct.keys())

        ## ensure that every required key from the schema is present
        if missing := real_type.__required_keys__ - keys:
            ks = ", ".join(sorted(repr(k) for k in missing))
            raise ValueError(f"Missing required field(s) for {real_type.__name__}: {ks}")

        ## ensure that there aren't any superfluous keys
        if closed_typed_dicts and (unexpected := keys - (real_type.__required_keys__ | real_type.__optional_keys__)):
            ks = ", ".join(sorted(repr(k) for k in unexpected))
            raise ValueError(f"Unexpected field(s) for {real_type.__name__}: {ks}")

        return dct  # we return the bare dict since TypedDict is just dict at runtime anyway

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

            return tuple(as_type(v, t, **kwargs) for v, t in zip(value, args))

        # otherwise, it's a variadic container
        vals = (as_type(v, inner_type, **kwargs) for v in value)

        if inspect.isabstract(real_type):
            # We can't cast to an abstract container, so just return the value as a list
            return list(vals)

        return real_type(vals)

    # handle NewType
    # note that T = NewType("T", S) means that T.__supertype__ will be S, and we will just cast to S
    if hasattr(to, "__supertype__"):
        return as_type(value, to.__supertype__, **kwargs)

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
