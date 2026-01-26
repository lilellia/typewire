# changelog

## 1.3.0

- Adds `as_string` utility function for getting a readable string representation of the given type_hint.

## 1.2.0

- Adds `as_type` support for `TypedDict`, including validation with `total`, `Required`, and `NotRequired`.
- Adds `as_type` support for recursive type hints via implementations for `typing.ForwardRef`.
- Adds the `evaluate_forward_ref` and `get_typed_dict_key_sets` utility functions.

## 1.1.0

- Adds `unwrap` as a utility function for stripping `Annotated`, `NewType`, `Union`.

## 1.0.1

- Corrects behaviour of `as_type(None, str | None)`. Prior to this update, the result was `"None"` (the string). It now correctly returns `None`.

## 1.0.0

- Adds support for `NewType`

## 0.1.0

- Initial release
