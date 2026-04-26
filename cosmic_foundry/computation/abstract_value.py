"""AbstractValue: opaque tensor placeholder for abstract interpretation passes."""

from __future__ import annotations

from typing import Any


class JitIncompatibleError(Exception):
    """Raised when a traced function materializes an abstract Tensor value.

    Any bool(), float(), or index conversion of a Tensor during a TracingBackend
    run indicates data-dependent Python control flow that breaks JIT compilation.
    The traceback identifies the exact algorithm line where the violation occurs.
    """


class AbstractValue:
    """Shape-carrying placeholder used by AbstractBackend in place of raw array data.

    Carries the tensor shape for shape propagation through algorithm code.
    Raises JitIncompatibleError on any attempt to extract a numeric value,
    which catches data-dependent Python control flow that would break JIT.

    Construct abstract Tensors via Tensor._wrap(AbstractValue(shape), backend)
    rather than through to_native, so that the shape is explicit.
    """

    __slots__ = ("shape",)

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def __bool__(self) -> bool:
        raise JitIncompatibleError(
            "bool() of an abstract Tensor: data-dependent Python control flow "
            "detected. Replace `if tensor:` / `bool(tensor)` with tensor "
            "operations that stay in tensor land."
        )

    def __float__(self) -> float:
        raise JitIncompatibleError(
            "float() of an abstract Tensor: value extracted into Python. "
            "Replace float(tensor) with tensor operations; use .item() only "
            "at explicit materialization boundaries."
        )

    def __index__(self) -> int:
        raise JitIncompatibleError(
            "int/index conversion of an abstract Tensor: value used as a "
            "Python index. Replace with tensor gather operations (take, etc.)."
        )

    def __getitem__(self, idx: Any) -> AbstractValue:
        """Support raw _data[idx] access from Tensor.__getitem__."""
        if isinstance(idx, int | AbstractValue):
            return AbstractValue(self.shape[1:] if self.shape else ())
        if isinstance(idx, slice):
            n = self.shape[0] if self.shape else 0
            length = len(range(*idx.indices(n)))
            return AbstractValue((length,) + self.shape[1:])
        return AbstractValue(self.shape[1:] if self.shape else ())

    def __repr__(self) -> str:
        return f"AbstractValue({self.shape})"


__all__ = ["AbstractValue", "JitIncompatibleError"]
