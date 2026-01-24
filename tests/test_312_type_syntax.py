import sys

import pytest

from typewire.caster import as_type


@pytest.mark.skipif(sys.version_info < (3, 12), reason="`type` keyword not available before 3.12")
def test_recursive_cast_for_type_keyword() -> None:
    type AnotherTree = list[int | AnotherTree]

    value = ["1", ["2", "3"], ["4", ["5"]]]
    expected = [1, [2, 3], [4, [5]]]
    assert as_type(value, AnotherTree) == expected


@pytest.mark.skipif(sys.version_info < (3, 12), reason="`type` keyword not available before 3.12")
def test_cast_to_3_12_type_keyword_alias() -> None:
    type AliasType = list[int]

    assert as_type(("1", "2", "3"), AliasType) == [1, 2, 3]
