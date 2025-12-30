from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest

from typewire.caster import as_type


@pytest.mark.parametrize(
    "value,to,expected",
    [
        ("123", int, 123),
        (123, str, "123"),
        ("123", float, 123.0),
        (123.0, int, 123),
        ("hello", str, "hello"),
        (b"hello", bytes, b"hello"),
    ],
)
def test_basic_scalars(value: Any, to: Any, expected: Any) -> None:
    assert as_type(value, to) == expected


def test_failed_cast() -> None:
    with pytest.raises(ValueError, match="invalid literal for int"):
        as_type("abc", int)


def test_transparent_int_false() -> None:
    with pytest.raises(ValueError, match="invalid literal for int"):
        as_type("1.0", int)


def test_transparent_int_true() -> None:
    assert as_type("1.0", int, transparent_int=True) == 1


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1.344", 1),
        ("4.22e3", 4220),
        ("7.3089e-3", 0),
    ],
)
def test_transparent_int_true_noninteger(value: str, expected: int) -> None:
    assert as_type(value, int, transparent_int=True) == expected


def test_semantic_bool_false() -> None:
    assert as_type("false", bool) is True


def test_semantic_bool_true() -> None:
    assert as_type("false", bool, semantic_bool=True) is False


@pytest.mark.parametrize(
    "value,union,expected",
    [
        ("1", int | float, 1),
        ("1", float | int, 1.0),
        ("abc", str | int, "abc"),
        ("abc", int | str, "abc"),
    ],
)
def test_union(value: str, union: Any, expected: Any) -> None:
    assert as_type(value, union) == expected


def test_union_failed_cast() -> None:
    class X:
        pass

    with pytest.raises(ValueError, match="Value 'abc' does not match any type in"):
        as_type("abc", X | int)


@pytest.mark.parametrize(
    "value,optional,expected",
    [
        ("abc", str | None, "abc"),
        (None, int | None, None),
    ],
)
def test_optional(value: Any, optional: Any, expected: Any) -> None:
    assert as_type(value, optional) == expected


def test_optional_failed_cast() -> None:
    class X:
        pass

    with pytest.raises(ValueError, match="Value 'abc' does not match any type in"):
        as_type("abc", X | None)


def test_literal_success() -> None:
    assert as_type("abc", Literal["abc", "def"]) == "abc"


def test_literal_failed_cast() -> None:
    with pytest.raises(ValueError, match="Value 'ghi' does not match any literal in"):
        as_type("ghi", Literal["abc", "def"])


def test_literal_failed_cast_for_wrong_type() -> None:
    with pytest.raises(ValueError, match="Value '80' does not match any literal in"):
        as_type("80", Literal[80, 443])


@pytest.mark.parametrize(
    "value,to,expected",
    [
        (["1", "2.5"], list[float], [1.0, 2.5]),
        (["1", "2.5"], list[str], ["1", "2.5"]),
        (["1", "2.5"], list[int | float], [1, 2.5]),
    ],
)
def test_simple_containers(value: Any, to: Any, expected: Any) -> None:
    assert as_type(value, to) == expected


def test_mapping() -> None:
    data = {"port": "8080", "timeout": "30.5"}
    expected = {"port": 8080, "timeout": 30.5}
    assert as_type(data, dict[str, int | float]) == expected


def test_mapping_from_tuples() -> None:
    data = [("port", "8080"), ("timeout", "30.5")]
    expected = {"port": 8080, "timeout": 30.5}
    assert as_type(data, dict[str, int | float]) == expected


def test_mapping_from_tuples_invalid() -> None:
    data = ["port", "8080", "timeout", "30.5"]
    with pytest.raises(ValueError, match="not a mapping"):
        as_type(data, dict[str, int])


def test_fixed_tuples() -> None:
    target = tuple[int, str, float]
    assert as_type(["1", "hi", "1.2"], target) == (1, "hi", 1.2)


def test_fixed_tuple_failed_cast() -> None:
    target = tuple[int, str, float]
    with pytest.raises(ValueError):
        as_type(["1", "hi"], target)


def test_variadic_tuple() -> None:
    target = tuple[int, ...]
    assert as_type(["1", "2", "3"], target) == (1, 2, 3)


def test_annotated() -> None:
    target = Annotated[int, "some metadata"]
    assert as_type("10", target) == 10


def test_deep_nested_annotated() -> None:
    target = Annotated[Annotated[Annotated[int, "metadata 3"], "metadata 2"], "metadata 1"]
    assert as_type("10", target) == 10


def test_deep_nested() -> None:
    to = dict[str, dict[str, dict[str, int | Path]]]
    data = {
        "a": {"b": {"c": "10", "d": "20"}},
        "e": {
            "f": {"g": "30", "h": "40", "i": "50", "j": "60"},
            "k": {"l": "70", "m": "80", "n": "90", "o": "100", "p": "/home/user/Documents"},
        },
    }
    expected = {
        "a": {"b": {"c": 10, "d": 20}},
        "e": {
            "f": {"g": 30, "h": 40, "i": 50, "j": 60},
            "k": {"l": 70, "m": 80, "n": 90, "o": 100, "p": Path("/home/user/Documents")},
        },
    }
    assert as_type(data, to) == expected


def test_custom_class() -> None:
    @dataclass
    class X:
        value: str

    assert as_type("abc", X) == X("abc")


def test_iterable_excludes_strings() -> None:
    assert as_type("abc", Iterable[str]) == "abc"


def test_abstract_mapping() -> None:
    assert as_type({"a": "1"}, Mapping[str, int]) == {"a": 1}


def test_abstract_sequence() -> None:
    assert as_type(["3", "7"], Iterable[int]) == [3, 7]
