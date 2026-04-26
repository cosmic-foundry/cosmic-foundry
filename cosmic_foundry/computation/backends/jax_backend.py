"""JAX backend for Tensor: all operations delegate to JAX/XLA."""

from __future__ import annotations

from typing import Any

import jax
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
    device:
        JAX device kind to place tensors on, e.g. ``"cpu"`` or ``"gpu"``.
        When ``None`` (the default) JAX's process-wide default device is used.
        Raises ``ValueError`` at construction time if the requested device kind
        is not available.
    """

    def __init__(self, dtype: Any = None, device: str | None = None) -> None:
        self._dtype = dtype
        if device is not None:
            devices = jax.devices(device)
            if not devices:
                raise ValueError(f"No JAX device of kind '{device}' is available")
            self._device: Any = devices[0]
        else:
            self._device = None

    def _maybe_place(self, arr: Any) -> Any:
        if self._device is not None:
            return jax.device_put(arr, self._device)
        return arr

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def to_native(self, data: Any) -> Any:
        """Convert Python scalar or nested sequence to a JAX array."""
        return self._maybe_place(jnp.asarray(data, dtype=self._dtype))

    def from_native(self, raw: Any) -> Any:
        """Convert JAX array to a Python scalar or nested list."""
        if raw.ndim == 0:
            return float(raw)
        return raw.tolist()

    def infer_shape(self, raw: Any) -> tuple[int, ...]:
        return tuple(raw.shape)

    def copy(self, raw: Any) -> Any:
        return self._maybe_place(jnp.array(raw))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def zeros(self, shape: tuple[int, ...]) -> Any:
        dt = self._dtype if self._dtype is not None else jnp.float64
        arr = jnp.array(0.0, dtype=dt) if not shape else jnp.zeros(shape, dtype=dt)
        return self._maybe_place(arr)

    def eye(self, n: int) -> Any:
        dt = self._dtype if self._dtype is not None else jnp.float64
        return self._maybe_place(jnp.eye(n, dtype=dt))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def flatten(self, raw: Any) -> list[float]:
        return [float(x) for x in raw.ravel()]

    def norm(self, raw: Any) -> Any:
        return jnp.linalg.norm(raw)

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

    # ------------------------------------------------------------------
    # Reductions and element-wise ops for JIT-clean algorithms
    # ------------------------------------------------------------------

    def abs(self, raw: Any) -> Any:
        return jnp.abs(raw)

    def reduce_max(self, raw: Any) -> Any:
        return jnp.max(raw)

    def argmax(self, raw: Any) -> Any:
        return jnp.argmax(raw)

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return jnp.where(cond, x, y)

    def lt(self, a: Any, b: Any) -> Any:
        return jnp.less(a, b)

    def le(self, a: Any, b: Any) -> Any:
        return jnp.less_equal(a, b)

    def gt(self, a: Any, b: Any) -> Any:
        return jnp.greater(a, b)

    def ge(self, a: Any, b: Any) -> Any:
        return jnp.greater_equal(a, b)

    def logical_not(self, raw: Any) -> Any:
        return jnp.logical_not(raw)

    def logical_or(self, a: Any, b: Any) -> Any:
        return jnp.logical_or(a, b)

    def arange(self, n: int) -> Any:
        return jnp.arange(n)

    def rdiv_scalar(self, s: float, raw: Any) -> Any:
        return jnp.divide(s, raw)

    def take(self, raw: Any, indices: Any) -> Any:
        return raw[indices]

    def get(self, raw: Any) -> Any:
        return raw.item()

    def sync(self, raw: Any) -> None:
        jax.block_until_ready(raw)

    def fori_loop(self, n: int, body_fn: Any, init_state: Any) -> Any:
        return jax.lax.fori_loop(0, n, body_fn, init_state)

    def while_loop(
        self,
        cond_fn: Any,
        body_fn: Any,
        init_state: Any,
    ) -> Any:
        """Ship the iteration loop to XLA via jax.lax.while_loop.

        cond_fn returns a 0-d bool Tensor; we unwrap to its raw JAX scalar so
        XLA evaluates the convergence test on-device on every iteration with
        no host roundtrip.  Tensor is registered as a JAX pytree (see
        cosmic_foundry.computation), so init_state and the body's return value
        are flattened/unflattened transparently.
        """
        return jax.lax.while_loop(
            cond_fun=lambda s: cond_fn(s)._value,
            body_fun=body_fn,
            init_val=init_state,
        )

    def fori_loop(
        self,
        n: int,
        body_fn: Any,
        init_state: Any,
    ) -> Any:
        """Ship a counted loop to XLA via jax.lax.fori_loop.

        body_fn(k, state) -> state where k is a traced device integer, so the
        entire n-iteration loop compiles to one XLA kernel.  Tensor pytree
        registration handles state flattening/unflattening transparently.
        """
        return jax.lax.fori_loop(0, n, body_fn, init_state)


__all__ = ["JaxBackend"]
