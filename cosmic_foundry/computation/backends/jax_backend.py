"""JAX backend for Tensor: all operations delegate to JAX/XLA."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


class JaxBackend:
    """Implements all Tensor operations using JAX arrays.

    JAX arrays are immutable; ``slice_set`` uses ``raw.at[idx].set(value)``
    and returns the updated array.  ``Tensor.__setitem__`` reassigns
    ``self._data`` with the result, so all mutation semantics are preserved
    at the ``Tensor`` level.

    float64 precision requires ``jax_enable_x64 = True``, which is set
    at module load time in ``cosmic_foundry.computation``.

    Parameters
    ----------
    dtype:
        Element dtype for ``zeros`` and ``eye``.  When ``None`` (the default)
        the dtype is inferred from the input data.
    """

    def __init__(self, dtype: Any = None) -> None:
        self._dtype = dtype

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def to_native(self, data: Any) -> Any:
        """Convert Python scalar or nested sequence to a JAX array."""
        return jnp.asarray(data, dtype=self._dtype)

    def from_native(self, raw: Any) -> Any:
        """Convert JAX array to a Python scalar or nested list."""
        if raw.ndim == 0:
            return float(raw)
        return raw.tolist()

    def infer_shape(self, raw: Any) -> tuple[int, ...]:
        return tuple(raw.shape)

    def copy(self, raw: Any) -> Any:
        return jnp.array(raw)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def zeros(self, shape: tuple[int, ...]) -> Any:
        dt = self._dtype if self._dtype is not None else jnp.float64
        if not shape:
            return jnp.array(0.0, dtype=dt)
        return jnp.zeros(shape, dtype=dt)

    def eye(self, n: int) -> Any:
        dt = self._dtype if self._dtype is not None else jnp.float64
        return jnp.eye(n, dtype=dt)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def flatten(self, raw: Any) -> list[float]:
        return [float(x) for x in raw.ravel()]

    def norm(self, raw: Any) -> float:
        return float(jnp.linalg.norm(raw))

    # ------------------------------------------------------------------
    # Element-wise arithmetic
    # ------------------------------------------------------------------

    def neg(self, a: Any) -> Any:
        return jnp.negative(a)

    def add(self, a: Any, b: Any) -> Any:
        return jnp.add(a, b)

    def sub(self, a: Any, b: Any) -> Any:
        return jnp.subtract(a, b)

    def mul_scalar(self, a: Any, s: float) -> Any:
        return jnp.multiply(a, s)

    def mul_elem(self, a: Any, b: Any) -> Any:
        return jnp.multiply(a, b)

    def div_scalar(self, a: Any, s: float) -> Any:
        return jnp.divide(a, s)

    def div_elem(self, a: Any, b: Any) -> Any:
        return jnp.divide(a, b)

    # ------------------------------------------------------------------
    # Contraction
    # ------------------------------------------------------------------

    def matmul(
        self,
        a: Any,
        b: Any,
        shape_a: tuple[int, ...],
        shape_b: tuple[int, ...],
    ) -> Any:
        return jnp.matmul(a, b)

    def einsum(
        self,
        spec: str,
        raws: list[Any],
        shapes: list[tuple[int, ...]],
    ) -> Any:
        return jnp.einsum(spec, *raws)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def diag(self, raw: Any, shape: tuple[int, ...]) -> Any:
        return jnp.diag(raw)

    def svd(self, raw: Any, shape: tuple[int, ...]) -> tuple[Any, Any, Any]:
        u, s, vt = jnp.linalg.svd(raw, full_matrices=False)
        return u, s, vt

    # ------------------------------------------------------------------
    # Slice indexing
    # ------------------------------------------------------------------

    def slice_get(self, raw: Any, idx: Any, shape: tuple[int, ...]) -> Any:
        return raw[idx]

    def slice_set(self, raw: Any, idx: Any, value: Any, shape: tuple[int, ...]) -> Any:
        """Return raw.at[idx].set(value); JAX arrays are immutable."""
        return raw.at[idx].set(value)


__all__ = ["JaxBackend"]
