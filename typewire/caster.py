from collections.abc import Mapping
from contextlib import suppress
import inspect
import sys
from typing import (
    Annotated,
    Any,
    ForwardRef,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
    Literal,
    TypedDict,
    TypeVar,
)

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required

if sys.version_info >= (3, 12):
    from typing import TypeAliasType

from .forward_refs import evaluate_forward_ref
from .identifier import is_iterable, is_mapping, is_union, TypeHint
from .typed_dict import get_typed_dict_key_sets


class _AsTypeKwargs(TypedDict):
    transparent_int: bool
    semantic_bool: bool
    closed_typed_dicts: bool
    _namespace: dict[str, Any] | None


def as_type(
    value: Any,
    to: TypeHint,
    *,
    transparent_int: bool = False,
    semantic_bool: bool = False,
    closed_typed_dicts: bool = False,
    _namespace: dict[str, Any] | None = None,
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
    # We can't cast to Any or an unbound TypeVar, so just return the value as-is
    if to is Any or isinstance(to, TypeVar):
        return value

    if _namespace is None:
        if (curr_frame := inspect.currentframe()) and (caller_frame := curr_frame.f_back):
            _namespace = {**caller_frame.f_globals, **caller_frame.f_locals}

    origin: Any = get_origin(to)
    args: Any = get_args(to)

    search = [to, *args] if origin else [to]
    for target in search:
        target_module = getattr(target, "__module__", None)
        if target_module and target_module not in ("typing", "builtins", "typing_extensions"):
            if target_module in sys.modules:
                # Merge the module dict, but let the captured namespace (from frame) keep priority
                _namespace = {**sys.modules[target_module].__dict__, **(_namespace or {})}
                break

    # Resolve situation when `to` is a str or ForwardRef
    if isinstance(to, (str, ForwardRef)):
        if _namespace:
            to = evaluate_forward_ref(to, namespace=_namespace)

    kwargs: _AsTypeKwargs = {
        "transparent_int": transparent_int,
        "semantic_bool": semantic_bool,
        "closed_typed_dicts": closed_typed_dicts,
        "_namespace": _namespace,
    }

    origin, args = get_origin(to), get_args(to)

    # reach into Annotated
    if origin in (Annotated, Required, NotRequired):
        to = get_args(to)[0]
        origin = get_origin(to)
        args = get_args(to)

    # reach into TypeAliasType (the 3.12 `type` keyword syntax)
    if sys.version_info >= (3, 12) and type(to) is TypeAliasType:
        to = to.__value__
        origin = get_origin(to)
        args = get_args(to)

    # handle unions
    if is_union(to):
        if value is None and type(None) in args:
            # if we're allowed to have None in the union, then return that
            return None

        for type_hint in args:
            if isinstance(type_hint, (str, ForwardRef)):
                print(f"DEBUG: type_hint: {type_hint!r} is still a ForwardRef")
            with suppress(ValueError, TypeError):
                return as_type(value, type_hint, **kwargs)
        else:
            reprs = ", ".join(repr(a) for a in args)
            raise ValueError(f"Value {value!r} does not match any type in {to}. Possible types: {reprs}")

    # handle literals
    if origin is Literal:
        if value in args:
            return value

        raise ValueError(f"Value {value!r} does not match any literal in {to}")

    # If `to` is a plain type (e.g., int), then origin is None. But we want something we can actually call.
    real_type = origin if origin is not None else to

    # handle mappings
    if is_mapping(real_type) and not is_typeddict(real_type):
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
    if is_typeddict(real_type):
        if not isinstance(value, Mapping):
            # input is a list of pairs like [("a", 1), ("b", 2)]
            try:
                value = dict(value)
            except (ValueError, TypeError):
                raise ValueError(f"Value {value!r} is not a mapping")

        annot = get_type_hints(real_type, globalns=_namespace, include_extras=True)
        key_sets = get_typed_dict_key_sets(real_type, _globalns=_namespace)

        # perform casting
        dct = {key: as_type(val, annot.get(key, Any), **kwargs) for key, val in value.items()}

        # perform validation
        keys = set(dct.keys())

        ## ensure that every required key from the schema is present
        if missing := key_sets.required - keys:
            ks = ", ".join(sorted(repr(k) for k in missing))
            raise ValueError(f"Missing required field(s) for {real_type.__name__}: {ks}")

        ## ensure that there aren't any superfluous keys
        if closed_typed_dicts and (unexpected := keys - key_sets.all()):
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
