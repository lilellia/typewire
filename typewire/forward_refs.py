import inspect
import sys
from typing import Any, ForwardRef, get_type_hints


def evaluate_forward_ref(ref: str | ForwardRef, /, *, namespace: dict[str, Any] | None = None) -> Any:
    if not isinstance(ref, (str, ForwardRef)):
        raise TypeError(f"Expected str or ForwardRef, got {type(ref)}")

    if namespace is None:
        if (current_frame := inspect.currentframe()) is not None and (caller_frame := current_frame.f_back) is not None:
            namespace = {**caller_frame.f_locals, **caller_frame.f_globals}

    # Wrap the reference in a dummy class so that we can use typing.get_type_hints.
    class _DummyClass:
        __annotations__ = {"ref": ref}

    try:
        hints = get_type_hints(_DummyClass, globalns=namespace)
        return hints["ref"]
    except Exception:
        # fall back to ForwardRef._evaluate
        # make sure we have an actual ForwardRef object
        ref = ref if isinstance(ref, ForwardRef) else ForwardRef(ref)

        if sys.version_info >= (3, 12):
            return ref._evaluate(
                globalns=namespace, localns=namespace, type_params=tuple(), recursive_guard=frozenset()
            )

        return ref._evaluate(globalns=namespace, localns=namespace, recursive_guard=frozenset())
