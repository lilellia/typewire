# typewire

A single-file utility to allow better runtime handling of types by providing a predictable way of transforming data into the shape of a given type hint.

## Why?

Python's standard library provides tools for describing types via type hints, but it doesn't provide a unified way of actually enforcing those type hints at runtime, even buried deep in the `typing` module.

Our goal is to allow for `x: T` to behave transparently as usual whilst also allowing the user to convert to that type, regardless of whether that is `x: int` or `x: float | None` or `x: dict[str, list[int | dict[float, User]]]`. Just like `int(x)` will (to the best of its ability) turn `x` into an `int`, `typewire.as_type(x, int)` will do the same, but with the added benefit of working on type hints that aren't callable (like `list[float]`).

```py
>>> from typewire import as_type

>>> as_type("3.2", float)
3.2

>>> as_type(78.1, int)
78

>>> as_type("3.2", int, transparent_int=True)
3

>>> as_type(["13.2", "18", "-1.2"], list[int | float])
[13.2, 18, -1.2]

>>> as_type([("a", "1"), ("b", "2"), ("c", "3"), ("z", "26")], dict[str, float])
{'a': 1.0, 'b': 2.0, 'c': 3.0, 'z': 26.0}

>>> from pathlib import Path
>>> data = {"logs": ["/tmp/app.log", "123"]}
>>> hint = dict[str, list[Path | int]]
>>> as_type(data, hint)
{'logs': [Path('/tmp/app.log'), 123]}
```

## Installation

`typewire` is supported on Python 3.10 and onward and can be easily installed with a package manager such as:

```bash
# using pip
$ pip install typewire

# using uv
$ uv add typewire
```

`typewire` does not have any additional dependencies.

## Documentation

### `TypeHint`

`TypeHint` is provided as a top-level alias for `typing.Any`.

### `is_union`, `is_mapping`, `is_iterable`

These three functions check whether a given type hint is a union type (e.g., `int | str | bytes`), a mapping type (e.g., `dict[str, Any]`), or an iterable type (`e.g., list[str]`).

Note that `is_iterable` specifically excludes `str` and `bytes`: `is_iterable(str) == False` as, while `str` does support iteration, for the purposes of type casting, it's not really an iterable/container type.

### `as_type`

The signature is

```py
def as_type(value: Any, to: TypeHint, *, transparent_int: bool = False, semantic_bool: False = False) -> Any:
  ...
```

In particular, it casts the given `value` to the given `to` type, regardless of whether `to` is:

```py

# a plain type
>>> as_type(3.2, int)
3

# typing.Literal, returning the value as-is if it's a valid entry
>>> as_type("abc", Literal["abc", "def"])
'abc'

>>> as_type("80", Literal[80, 443])
ValueError(...)

# a union type, casting to the first valid type
>>> as_type("3", float | int)
3.0

>>> as_type("3", int | float)
3

# an optional type
>>> as_type(43, int | None)
43

>>> as_type(None, int | None)
None

# a mapping type
>>> as_type({"a": "1", "b": "2.0"}, dict[str, float])
{'a': 1.0, 'b': 2.0}

# a container/iterable type
>>> as_type([1.2, -3, 449], list[str])
['1.2', '-3', '449']

>>> as_type([1.2, -3, 449], tuple[str, ...])
('1.2', '-3', '449')

# typing.Annotated, treating it as the bare type
>>> as_type("3", Annotated[int, "some metadata"])
3

# an abstract collections.abc.Iterable/Mapping, cast as concrete list/dict
>>> as_type([1.2, -3, 449], Iterable[str])
['1.2', '-3', '449']

>>> as_type({"a": "1", "b": "2.0"}, Mapping[str, float])
{'a': 1.0, 'b': 2.0}

# ...unless it's a string being cast as Iterable[str]
>>> as_type("hello world", Iterable[str])
'hello world'
```

On a failure, `ValueError` is raised.

#### `transparent_int`

This flag (default = False) allows for a nonstrict cast to `int`.

```py
>>> int("3.2", int)
ValueError # invalid literal for int() with base 10: '3.2'

>>> as_type("3.2", int)
ValueError # invalid literal for int() with base 10: '3.2'

>>> as_type("3.2", int, transparent_int = True)
3
```

In practice, this flag results in a call of `int(float(value))` instead of just `int(value)`.

#### `semantic_bool`

This flag (default = False) allows for a nonstrict cast to `bool`.

```py
>>> bool("false")  # non-empty string
True

>>> as_type("false", bool)
True

>>> as_type("false", bool, semantic_bool = True)
False
```

In practice, if `value` is a string and is one of `["false", "no", "0", "off"]` (case-insensitive), then it will be cast as `False` with this flag enabled.
