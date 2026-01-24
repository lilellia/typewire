# typewire

A utility to allow better runtime handling of types by providing a predictable way of transforming data into the shape of a given type hint.

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

`typewire` does not have any additional dependencies except for, on Python 3.10 only, `typing_extensions`.

## Documentation

### `TypeHint`

`TypeHint` is provided as a top-level alias for `typing.Any`.

### `is_union`, `is_mapping`, `is_iterable`

These three functions check whether a given type hint is a union type (e.g., `int | str | bytes`), a mapping type (e.g., `dict[str, Any]`), or an iterable type (`e.g., list[str]`).

Note that `is_iterable` specifically excludes `str` and `bytes`: `is_iterable(str) == False` as, while `str` does support iteration, for the purposes of type casting, it's not really an iterable/container type.

### `as_type`

The signature is

```py
def as_type(value: Any, to: TypeHint, *, transparent_int: bool = False, semantic_bool: bool = False, closed_typed_dicts: bool = False) -> Any:
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

# even TypedDict
>>> class Point(TypedDict):
...     x: float
...     y: float

>>> as_type({"x": "1.0", "y": "2.0"}, Point)  # note that TypedDict collapses to regular dict at runtime
{'x': 1.0, 'y': 2.0}

>>> as_type({"x": "1.0"}, Point)  # note the missing required 'y' field
ValueError("Missing required field(s) for Point: 'y'")

>>> as_type({"x": "1.0", "y": "2.0", "z": "3.0"}, Point)  # note the extra 'z' field, which is left alone...
{'x': 1.0, 'y': 2.0, 'z': '3.0'}

>>> as_type({"x": "1.0", "y": "2.0", "z": "3.0"}, Point, closed_typed_dicts=True)  # ...unless you tell typewire that TypedDicts should be closed
ValueError("Unexpected field(s) for Point: 'z'")

# a container/iterable type
>>> as_type([1.2, -3, 449], list[str])
['1.2', '-3', '449']

>>> as_type([1.2, -3, 449], tuple[str, ...])
('1.2', '-3', '449')

# even if it's just a blank generic continer: it'll act like T[Any]
>>> as_type(["a", 3.2, None, [1, "a"]], tuple)
('a', 3.2, None, [1, 'a'])

# typing.Annotated, treating it as the bare type
>>> as_type("3", Annotated[int, "some metadata"])
3

# a typing.NewType, treating it as the supertype
>>> UserId = NewType("UserId", int)
>>> as_type("3", UserId)
3
>>> type(as_type("3", UserId))  # unfortunately, the UserId type doesn't exist at runtime
<class 'int'>

# a recursive type
>>> class Node(TypedDict):
...     data: int
...     next: NotRequired[Node]

>>> as_type({"data": "12", "next": {"data": "17"}}, Node)
{'data': 12, 'next': {'data': 17}}

>>> type Tree = list[int | Tree]
>>> as_type(["1", ["2", "3"], ["4", ["5"]]], Tree)
[1, [2, 3], [4, [5]]]

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
>>> int("3.2")
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

#### `closed_typed_dicts`

When `to` is (or contains) a TypedDict, this determines whether additional keys beyond the TypedDict's schema are allowed. With `closed_typed_dicts=True`, additional keys will raise a `ValueError`. That is,

```py
>>> class Point(TypedDict):
...     x: float
...     y: float

>>> as_type({"x": "1.0", "y": "2.0", "z": "3.0"}, Point, closed_typed_dicts=False)
{'x': 1.0, 'y': 2.0, 'z': '3.0'}

>>> as_type({"x": "1.0", "y": "2.0", "z": "3.0"}, Point, closed_typed_dicts=True)
ValueError("Unexpected field(s) for Point: 'z'")
```

### `unwrap`

`unwrap` recursively removes `Annotated`, `NewType`, and `Union` layers, returning a list of component types. Note that it does *not* unwrap other containers, such as `list`. This is because `unwrap` is working to identify what the type "is", rather to find all of the structural components. From that perspective, `list[T]` is itself a leaf: the type represents a list.

```py
# bare types are just themselves
>>> unwrap(int)
[int]

>>> unwrap(list[int])
[list[int]]

# note that None is interpreted to NoneType, i.e., type(None)
>>> unwrap(None)
[<class 'NoneType'>]

# union types return their components in order
>>> unwrap(int | str)
[int, str]
>>> unwrap(str | int)
[str, int]

# resulting types are deduplicated
>>> unwrap(int | str | int)
[int, str]

# also works with old-style Optional
>>> unwrap(Optional[int])
[int, <class 'NoneType'>]

# annotated layer gets removed
>>> unwrap(Annotated[int, "some metadata"])
[int]

# NewType gets unwrapped, returning the supertype
>>> UserId = NewType("UserId", int)
>>> unwrap(UserId)
[int]

# unwrap can handle matroyska dolls of nesting
# resulting order is depth-first
>>> T1, T2, T3 = TypeVar("T1"), TypeVar("T2"), TypeVar("T3")
>>> D1 = NewType("D1", T1)
>>> matroyska_doll = Optional[Annotated[T1 | Annotated[T2 | D1 | None | Annotated[T3 | T1, "level 2"], "level 1"], "level 2"]]
>>> unwrap(matroyska_doll)
[T1, T2, <class 'NoneType'>, T3]  # D1 -> T1, so it doesn't appear in the result, nor does the final T1
```

### `get_typed_dict_key_sets`

A utility function that provides the keys (required and optional) for a TypedDict. Normally, you'd use `t.__required_keys__` and `t.__optional_keys__`, but this doesn't work when the type hints (including `Required[T]` and `NotRequired[T]`) have been coerced to strings via `from __future__ import annotations`. This function will provide them even in that case.

The return type is a NamedTuple with `required` and `optional` attributes, both of which are `frozenset[str]`, in alignment with the usual `__required_keys__` and `__optional_keys__` types:

```py
>>> from __future__ import annotations
>>> from typing import TypedDict, NotRequired
>>> from typewire import get_typed_dict_key_sets

>>> class Point(TypedDict):
...     x: float
...     y: float
...     z: NotRequired[float]

# This contains 'z' even though it's explicitly NotRequired!
# This is because the hint (due to the `from __future__ import annotations`)
# is the literal string "NotRequired[float]", which doesn't get parsed.
>>> Point.__required_keys__
frozenset({'x', 'y', 'z'})

# And this is empty for the same reason.
>>> Point.__optional_keys__
frozenset()

# typewire provides the parsing, though:
>>> get_typed_dict_key_sets(Point)
TypedDictKeySets(required=frozenset({'x', 'y'}), optional=frozenset({'z'}))

>>> required, optional = get_typed_dict_key_sets(Point)
>>> required
frozenset({'x', 'y'})
>>> optional
frozenset({'z'})
```

### `evaluate_forward_ref`

A utility function that works to evaluate a ForwardRef (or str), coercing it into a usable runtime object.

The signature is `def evaluate_forward_ref(ref: str | ForwardRef, /, *, namespace: dict[str, Any] | None = None) -> Any`. If `namespace` is None, then it pulls the globals and locals from the caller's frame, if available.

```py
class X:
    pass

print(evaluate_forward_ref(typing.ForwardRef("X")))  # <class '__main__.X'>
assert evaluate_forward_ref(typing.ForwardRef("X")) is X
```

You wouldn't normally create a ForwardRef object manually, but it's automatically generated any time that a type hint refers to something that hasn't been defined yet (either for recursive types or for another type later in the file). As such, this is primarily an internal function, but it's provided for convenience for anyone who might need it.
