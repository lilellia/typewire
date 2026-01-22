from typing import Annotated, NewType, Optional, TypeVar, Union

import pytest

from typewire import TypeHint, unwrap

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
Derived = NewType("Derived", T1)
ConcreteDerived = NewType("ConcreteDerived", list[str | int | dict[float, int]])


class UnhashableType:
    __hash__ = None

    def __eq__(self, other):
        return self is other


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (int, [int]),
        (str, [str]),
        (None, [type(None)]),
        (T1, [T1]),
        (list[int], [list[int]]),
        (dict[str, int], [dict[str, int]]),
        (list[T1 | T2], [list[T1 | T2]]),
        (UnhashableType, [UnhashableType]),
    ],
)
def test_unwrap_base_type(type_hint: TypeHint, expected: list[TypeHint]) -> None:
    assert unwrap(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (T1, [T1]),
        (Annotated[T1, "level 1"], [T1]),
        (Annotated[Annotated[T1, "level 1"], "level 2"], [T1]),
        (Annotated[Annotated[Annotated[T1, "level 1"], "level 2"], "level 3"], [T1]),
        (Annotated[Annotated[Annotated[Annotated[T1, "level 1"], "level 2"], "level 3"], "level 4"], [T1]),
    ],
)
def test_unwrap_levels_of_nested_annotated(type_hint: TypeHint, expected: list[TypeHint]) -> None:
    assert unwrap(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        # 3.10 unions
        (T1 | T2, [T1, T2]),
        (int | str, [int, str]),
        (T1 | T2 | T3, [T1, T2, T3]),
        (int | str | float, [int, str, float]),
        (int | str | float | None, [int, str, float, type(None)]),
        # pre-3.10 unions
        (Union[T1, T2], [T1, T2]),
        (Union[int, str], [int, str]),
        (Union[T1, T2, T3], [T1, T2, T3]),
        (Union[int, str, float], [int, str, float]),
        (Union[int, str, float, None], [int, str, float, type(None)]),
    ],
)
def test_unwrap_basic_unions(type_hint: TypeHint, expected: list[TypeHint]) -> None:
    assert unwrap(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (T1 | T1, [T1]),
        (T1 | T2 | T1, [T1, T2]),
        (T1 | T2 | T3 | T1, [T1, T2, T3]),
        (T1 | T2 | T3 | T1 | T2, [T1, T2, T3]),
    ],
)
def test_unwrap_duplicate_unions(type_hint: TypeHint, expected: list[TypeHint]) -> None:
    assert unwrap(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (Optional[T1], [T1, type(None)]),
        (Optional[int], [int, type(None)]),
    ],
)
def test_unwrap_old_style_optionals(type_hint: TypeHint, expected: list[TypeHint]) -> None:
    assert unwrap(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (Derived, [T1]),
        (ConcreteDerived, [list[str | int | dict[float, int]]]),
        (Derived | T1, [T1]),
        (ConcreteDerived | list[str | int | dict[float, int]], [list[str | int | dict[float, int]]]),
        (Derived | T2, [T1, T2]),
        (ConcreteDerived | list[str | int | dict[float, int]] | float, [list[str | int | dict[float, int]], float]),
    ],
)
def test_unwrap_new_type(type_hint: TypeHint, expected: list[TypeHint]) -> None:
    assert unwrap(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        (Annotated[T1 | Optional[T2 | Annotated[T3 | None, "level 2"]], "level 1"], [T1, T2, T3, type(None)]),
        (
            Optional[
                Annotated[T1 | Annotated[T2 | Derived | None | Annotated[T3 | T1, "level 2"], "level 1"], "level 2"]
            ],
            [T1, T2, type(None), T3],
        ),
    ],
)
def test_unwrap_matroyska_doll(type_hint: TypeHint, expected: list[TypeHint]) -> None:
    assert unwrap(type_hint) == expected


def test_unwrap_unhashable_types() -> None:
    t1, t2, t3 = UnhashableType(), UnhashableType(), UnhashableType()
    assert unwrap(Union[t1, t2, t3, t1]) == [t1, t2, t3]
