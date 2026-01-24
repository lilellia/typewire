import sys
from typing import Any, get_origin, get_type_hints, is_typeddict, NamedTuple

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required


class TypedDictKeySets(NamedTuple):
    required: frozenset[str]
    optional: frozenset[str]

    def all(self) -> frozenset[str]:
        return self.required | self.optional


def get_typed_dict_key_sets(typed_dict: type, *, _globalns: dict[str, Any] | None = None) -> TypedDictKeySets:
    if not is_typeddict(typed_dict):
        raise TypeError(f"{typed_dict} is not a TypedDict")

    required: set[str] = set()
    optional: set[str] = set()

    is_total = getattr(typed_dict, "__total__", True)
    for key, hint in get_type_hints(typed_dict, globalns=_globalns, include_extras=True).items():
        origin = get_origin(hint)

        if is_total:
            s = optional if origin is NotRequired else required
        else:
            s = required if origin is Required else optional

        s.add(key)

    return TypedDictKeySets(required=frozenset(required), optional=frozenset(optional))
