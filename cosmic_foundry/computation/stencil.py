"""Stencil: pointwise stencil ABC for structured-grid Functions."""

from __future__ import annotations

import functools
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.computation.descriptor import Extent, _checked_bounds
from cosmic_foundry.theory.function import Function


class Stencil(Function):
    """ABC for pointwise stencil operators on structured grids.

    Function:
        domain   — one or more array-valued fields on Ω_h ⊆ ℤⁿ
        codomain — an array-valued field on Ω_h^int ⊆ Ω_h
        operator — pointwise application of _fn over an Extent

    Subclasses must define:
    - ``radii: tuple[int, ...]`` — stencil half-widths per axis
    - ``_fn(*field_arrays, *index_meshgrids) -> scalar`` — pointwise kernel

    ``execute(*field_arrays, extent=...)`` is provided automatically.
    """

    radii: tuple[int, ...]

    @abstractmethod
    def _fn(self, *args: Any) -> Any:
        """Pointwise stencil kernel evaluated at a single grid point."""

    def execute(self, *field_arrays: Any, extent: Extent) -> Any:
        _validate_halo_access(extent, self.radii, field_arrays)
        return _make_jit_kernel(cast(Any, self), extent)(*field_arrays)


@functools.lru_cache(maxsize=256)
def _make_jit_kernel(fn: Any, extent: Any) -> Callable[..., Any]:
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


def _validate_halo_access(
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


__all__ = ["Stencil"]
