from __future__ import annotations

from collections.abc import Callable, Iterable
import sys
from typing import Annotated, Literal, ParamSpec, TypedDict, TypeVar

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required

import pytest

from typewire import as_string, TypeHint

T = TypeVar("T")
P = ParamSpec("P")


@pytest.mark.parametrize(
    "type_hint,expected",
    [(int, "int"), (str, "str"), (bytes, "bytes"), (None, "None"), (T, "T")],
)
def test_basic_scalar_types(type_hint: TypeHint, expected: str) -> None:
    assert as_string(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (int | str, "int | str"),
        (str | int, "str | int"),
        (int | str | None, "int | str | None"),
        (float | list | dict, "float | list | dict"),
        (list[int] | None | str, "list[int] | None | str"),
        (set[T], "set[T]"),  # type: ignore[valid-type]
    ],
)
def test_basic_union_types(type_hint: TypeHint, expected: str) -> None:
    assert as_string(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (list[int], "list[int]"),
        (dict[str, int], "dict[str, int]"),
        (set[int], "set[int]"),
        (list[int], "list[int]"),
        (dict[str, int], "dict[str, int]"),
        (Iterable[str], "Iterable[str]"),
        (tuple[int], "tuple[int]"),
        (tuple[int, int], "tuple[int, int]"),
        (tuple[int, int, int], "tuple[int, int, int]"),
        (tuple[int, ...], "tuple[int, ...]"),
        (tuple[T, ...], "tuple[T, ...]"),  # type: ignore[valid-type]
    ],
)
def test_container_types(type_hint: TypeHint, expected: str) -> None:
    assert as_string(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (Annotated, "Annotated"),
        (Annotated[str, "some metadata"], "Annotated[str, ...]"),
        (Annotated[int, "some metadata"], "Annotated[int, ...]"),
        # Python internally flattens nested Annotated types, so we can't render this in any nested way
        # Annotated[Annotated[T, M1], M2] is internally identical to Annotated[T, M1, M2]
        (Annotated[Annotated[bytes, "some metadata"], "more metadata"], "Annotated[bytes, ...]"),
        (Literal["a", "b", 17], "Literal['a', 'b', 17]"),
        (Literal, "Literal"),
        (NotRequired[int], "NotRequired[int]"),
        (Required[str], "Required[str]"),
    ],
)
def test_labeled_types(type_hint: TypeHint, expected: str) -> None:
    assert as_string(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (list[int] | float | dict[str, list[int | float]], "list[int] | float | dict[str, list[int | float]]"),
        (Iterable[int | float | Annotated[str, "some metadata"]], "Iterable[int | float | Annotated[str, ...]]"),
        (dict[str, dict[str, dict[str, dict[str, int]]]], "dict[str, dict[str, dict[str, dict[str, int]]]]"),
    ],
)
def test_nested_types(type_hint: TypeHint, expected: str) -> None:
    assert as_string(type_hint) == expected


def test_custom_type() -> None:
    class CustomType:
        pass

    assert as_string(CustomType) == "CustomType"


def test_typed_dict() -> None:
    class Data(TypedDict):
        key1: int
        key2: Required[str]
        key3: NotRequired[int]
        key4: list[int | float]

    expected = (
        "Data[key1: Required[int], key2: Required[str], key3: NotRequired[int], key4: Required[list[int | float]]]"
    )
    assert as_string(Data) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (Callable, "Callable"),
        (Callable[[], int], "Callable[[], int]"),
        (Callable[[int], str], "Callable[[int], str]"),
        (Callable[..., int], "Callable[..., int]"),
        (Callable[[int | list[int]], str | None], "Callable[[int | list[int]], str | None]"),
        (Callable[P, int | bytes], "Callable[P, int | bytes]"),
    ],
)
def test_callable(type_hint: TypeHint, expected: str) -> None:
    assert as_string(type_hint) == expected
