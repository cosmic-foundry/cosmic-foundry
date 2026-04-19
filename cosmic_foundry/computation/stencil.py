"""Stencil kernel execution engine.

``execute_pointwise`` applies a stencil Function over a structured-grid
Extent using JAX JIT compilation.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.computation.descriptor import (
    Extent,
    _checked_bounds,
)


def execute_pointwise(
    fn: Any,
    extent: Extent,
    *field_arrays: Any,
) -> Any:
    """Apply a stencil fn over extent with JAX JIT and input validation.

    ``fn`` must be hashable (for JIT caching) and expose:

    - ``_fn(*field_arrays, *index_meshgrids) -> scalar``
    - ``radii: tuple[int, ...]``
    """
    _validate_region_access(extent, fn.radii, field_arrays)
    return _make_jit_kernel(cast(Any, fn), extent)(*field_arrays)


@functools.lru_cache(maxsize=256)
def _make_jit_kernel(fn: Any, extent: Any) -> Callable[..., Any]:
    """Return a cached JIT-compiled kernel for *(fn, extent)*."""

    @jax.jit
    def apply(*jit_inputs: Any) -> Any:
        indices = _region_indices(extent)
        return fn._fn(*jit_inputs, *indices)

    return cast(Callable[..., Any], apply)


def _region_indices(extent: Extent) -> tuple[jax.Array, ...]:
    axes = []
    for axis_slice in extent.slices:
        start, stop = _checked_bounds(axis_slice)
        axes.append(jnp.arange(start, stop))
    return tuple(jnp.meshgrid(*axes, indexing="ij"))


def _validate_region_access(
    extent: Extent,
    radii: tuple[int, ...],
    inputs: tuple[Any, ...],
) -> None:
    required = extent.expand(radii)
    for input_array in inputs:
        if not hasattr(input_array, "shape"):
            msg = "Function inputs must expose a shape"
            raise TypeError(msg)
        shape = tuple(int(axis_size) for axis_size in input_array.shape)
        if len(shape) < required.ndim:
            msg = "Function input rank is smaller than the extent rank"
            raise ValueError(msg)
        for axis, axis_slice in enumerate(required.slices):
            start, stop = _checked_bounds(axis_slice)
            if start < 0 or stop > shape[axis]:
                msg = "Function extent plus radii exceeds input bounds"
                raise ValueError(msg)


__all__ = ["execute_pointwise"]
