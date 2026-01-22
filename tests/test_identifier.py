from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, Optional, Union

import pytest

from typewire import is_iterable, is_mapping, is_union, TypeHint


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        # bare types
        (None, False),
        (int, False),
        (object, False),
        # other weird types
        (Literal["a", "b"], False),
        (Annotated[int, "some metadata"], False),
        (Annotated[int | str, "some metadata"], False),
        # new 3.10 unions
        (int | float, True),
        (int | float | str, True),
        (int | None, True),
        # old pre-3.10 unions
        (Union[int, float], True),
        (Union[int, float, str], True),
        (Union[int, None], True),
        (Optional[int], True),
    ],
)
def test_is_union(type_hint: TypeHint, expected: bool) -> None:
    assert is_union(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        # bare types
        (None, False),
        (int, False),
        # wrapper types
        (Annotated[int, "some metadata"], False),
        (Annotated[dict, "some metadata"], False),
        (dict[str, int] | dict[int, str], False),
        # actual maps
        (dict, True),
        (dict[str, int], True),
        (dict[int, str], True),
        (Mapping[str, int], True),
    ],
)
def test_mapping(type_hint: TypeHint, expected: bool) -> None:
    assert is_mapping(type_hint) == expected


@pytest.mark.parametrize(
    "type_hint,expected",
    [
        # bare types
        (None, False),
        (int, False),
        (object, False),
        # wrapper types
        (Annotated[int, "some metadata"], False),
        (Annotated[list, "some metadata"], False),
        (list[int] | dict[str, int], False),
        # containers
        (list, True),
        (dict, True),
        (set, True),
        (list[int], True),
        (dict[str, int], True),
        (Iterable[str], True),
        # check exclusion for str and bytes
        (str, False),
        (bytes, False),
    ],
)
def test_is_iterable(type_hint: TypeHint, expected: bool) -> None:
    assert is_iterable(type_hint) == expected
