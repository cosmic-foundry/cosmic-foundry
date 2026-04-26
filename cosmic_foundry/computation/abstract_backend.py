"""AbstractBackend: shape-propagating backend for abstract interpretation passes."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.computation.abstract_value import (
    AbstractValue,
    JitIncompatibleError,
)


def _infer_shape(data: Any) -> tuple[int, ...]:
    """Infer tensor shape from a Python scalar or nested list."""
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _infer_shape(data[0])


def _broadcast(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
    """Return broadcast shape; 0-d (empty tuple) broadcasts with anything."""
    if not s1:
        return s2
    if not s2:
        return s1
    return s1 if len(s1) >= len(s2) else s2


def _slice_output_shape(idx: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the output shape of shape[idx] for slice-containing indices."""
    if isinstance(idx, slice):
        n = shape[0] if shape else 0
        length = len(range(*idx.indices(n)))
        return (length,) + shape[1:]
    if isinstance(idx, tuple):
        result: list[int] = []
        remaining = list(shape)
        for i in idx:
            if isinstance(i, int):
                if remaining:
                    remaining.pop(0)
            elif isinstance(i, slice):
                if remaining:
                    n = remaining.pop(0)
                    result.append(len(range(*i.indices(n))))
        result.extend(remaining)
        return tuple(result)
    # Dynamic index (AbstractValue or other): drops first dimension.
    return shape[1:] if shape else ()


def _matmul_output_shape(
    shape_a: tuple[int, ...], shape_b: tuple[int, ...]
) -> tuple[int, ...]:
    r1, r2 = len(shape_a), len(shape_b)
    if r1 == 1 and r2 == 1:
        return ()
    if r1 == 1 and r2 == 2:
        return (shape_b[1],)
    if r2 == 1:
        return shape_a[:-1]
    return shape_a[:-1] + (shape_b[-1],)


def _einsum_output_shape(spec: str, shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    spec = spec.replace(" ", "")
    lhs, out_spec = spec.split("->")
    in_specs = lhs.split(",")
    sizes: dict[str, int] = {}
    for s, shape in zip(in_specs, shapes, strict=False):
        for pos, ch in enumerate(s):
            sizes[ch] = shape[pos]
    return tuple(sizes[ch] for ch in out_spec)


class AbstractBackend:
    """Backend that propagates AbstractValues through all tensor operations.

    Every Backend protocol method returns an AbstractValue with the correct
    output shape rather than performing numeric computation.  Subclasses
    override _charge(flops) to accumulate cost (ProfilingBackend) or leave
    it as a no-op (TracingBackend).

    item() raises JitIncompatibleError, so any algorithm that calls
    float(tensor) or bool(tensor) is caught immediately during a tracing run.
    """

    def _charge(self, flops: int) -> None:
        """Called for each operation with its FLOP count. No-op by default."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def to_native(self, data: Any) -> AbstractValue:
        return AbstractValue(_infer_shape(data))

    def from_native(self, raw: Any) -> Any:
        return raw

    def infer_shape(self, raw: Any) -> tuple[int, ...]:
        if isinstance(raw, AbstractValue):
            return raw.shape
        return ()

    def copy(self, raw: Any) -> AbstractValue:
        return AbstractValue(raw.shape if isinstance(raw, AbstractValue) else ())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def zeros(self, shape: tuple[int, ...]) -> AbstractValue:
        return AbstractValue(shape)

    def eye(self, n: int) -> AbstractValue:
        return AbstractValue((n, n))

    # ------------------------------------------------------------------
    # Materialization boundary — always raises
    # ------------------------------------------------------------------

    def item(self, raw: Any) -> Any:
        raise JitIncompatibleError(
            "item() called inside a traced function: explicit materialization "
            "is not permitted in JIT-compiled code."
        )

    # ------------------------------------------------------------------
    # Utilities that return Python values — raise in traced context
    # ------------------------------------------------------------------

    def flatten(self, raw: Any) -> list[float]:
        raise JitIncompatibleError("flatten() materialises tensor values.")

    def norm(self, raw: Any) -> float:
        raise JitIncompatibleError("norm() materialises a tensor value.")

    # ------------------------------------------------------------------
    # Element-wise arithmetic
    # ------------------------------------------------------------------

    def neg(self, a: Any) -> AbstractValue:
        return AbstractValue(a.shape)

    def add(self, a: Any, b: Any) -> AbstractValue:
        return AbstractValue(_broadcast(a.shape, b.shape))

    def sub(self, a: Any, b: Any) -> AbstractValue:
        return AbstractValue(_broadcast(a.shape, b.shape))

    def mul_scalar(self, a: Any, s: float) -> AbstractValue:
        return AbstractValue(a.shape)

    def mul_elem(self, a: Any, b: Any) -> AbstractValue:
        return AbstractValue(_broadcast(a.shape, b.shape))

    def div_scalar(self, a: Any, s: float) -> AbstractValue:
        return AbstractValue(a.shape)

    def div_elem(self, a: Any, b: Any) -> AbstractValue:
        return AbstractValue(_broadcast(a.shape, b.shape))

    def rdiv_scalar(self, s: float, raw: Any) -> AbstractValue:
        return AbstractValue(raw.shape)

    def abs(self, raw: Any) -> AbstractValue:
        return AbstractValue(raw.shape)

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def lt(self, a: Any, b: Any) -> AbstractValue:
        a_s = a.shape if isinstance(a, AbstractValue) else ()
        b_s = b.shape if isinstance(b, AbstractValue) else ()
        return AbstractValue(_broadcast(a_s, b_s))

    def le(self, a: Any, b: Any) -> AbstractValue:
        return self.lt(a, b)

    def gt(self, a: Any, b: Any) -> AbstractValue:
        return self.lt(a, b)

    def ge(self, a: Any, b: Any) -> AbstractValue:
        return self.lt(a, b)

    def where(self, cond: Any, x: Any, y: Any) -> AbstractValue:
        c_s = cond.shape if isinstance(cond, AbstractValue) else ()
        x_s = x.shape if isinstance(x, AbstractValue) else ()
        y_s = y.shape if isinstance(y, AbstractValue) else ()
        return AbstractValue(_broadcast(_broadcast(c_s, x_s), y_s))

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def reduce_max(self, raw: Any) -> AbstractValue:
        return AbstractValue(())

    def argmax(self, raw: Any) -> AbstractValue:
        return AbstractValue(())

    # ------------------------------------------------------------------
    # Contraction
    # ------------------------------------------------------------------

    def matmul(
        self,
        a: Any,
        b: Any,
        shape_a: tuple[int, ...],
        shape_b: tuple[int, ...],
    ) -> AbstractValue:
        return AbstractValue(_matmul_output_shape(shape_a, shape_b))

    def einsum(
        self,
        spec: str,
        raws: list[Any],
        shapes: list[tuple[int, ...]],
    ) -> AbstractValue:
        return AbstractValue(_einsum_output_shape(spec, shapes))

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def diag(self, raw: Any, shape: tuple[int, ...]) -> AbstractValue:
        return AbstractValue((min(shape),))

    def svd(
        self, raw: Any, shape: tuple[int, ...]
    ) -> tuple[AbstractValue, AbstractValue, AbstractValue]:
        m, n = shape
        k = min(m, n)
        return AbstractValue((m, k)), AbstractValue((k,)), AbstractValue((k, k))

    # ------------------------------------------------------------------
    # Slice indexing
    # ------------------------------------------------------------------

    def slice_get(self, raw: Any, idx: Any, shape: tuple[int, ...]) -> AbstractValue:
        return AbstractValue(_slice_output_shape(idx, shape))

    def slice_set(
        self, raw: Any, idx: Any, value: Any, shape: tuple[int, ...]
    ) -> AbstractValue:
        return AbstractValue(shape)

    # ------------------------------------------------------------------
    # Gather / scatter
    # ------------------------------------------------------------------

    def take(self, raw: Any, indices: Any) -> AbstractValue:
        idx_shape = indices.shape if isinstance(indices, AbstractValue) else ()
        return AbstractValue(idx_shape)

    def arange(self, n: int) -> AbstractValue:
        return AbstractValue((n,))


__all__ = ["AbstractBackend"]
