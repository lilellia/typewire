from __future__ import annotations

import sys
from typing import Any, TypeAlias, TypedDict, Union

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

from typewire import as_type


class Node(TypedDict):
    value: int
    next: NotRequired[Node]


def test_cast_to_recursive_type() -> None:
    value = {"value": "12", "next": {"value": "17"}}
    assert as_type(value, Node) == Node(value=12, next=Node(value=17))


Tree: TypeAlias = list[Union[int, "Tree"]]


def test_cast_to_recursive_list() -> None:
    value = ["1", ["2", "3"], ["4", ["5"]]]
    expected = [1, [2, 3], [4, [5]]]
    assert as_type(value, Tree) == expected


class JSONNode(TypedDict):
    data: Any
    children: NotRequired[dict[str, JSONNode]]


def test_cast_to_deep_recursive_structure() -> None:
    value = {
        "data": "root",
        "children": {
            "child_1": {"data": 1, "children": {}},
            "child_2": {"data": 2, "children": {"child_2_1": {"data": 21, "children": {}}}},
        },
    }
    result = as_type(value, JSONNode)
    assert result == JSONNode(
        data="root",
        children={
            "child_1": JSONNode(data=1, children={}),
            "child_2": JSONNode(data=2, children={"child_2_1": JSONNode(data=21, children={})}),
        },
    )
