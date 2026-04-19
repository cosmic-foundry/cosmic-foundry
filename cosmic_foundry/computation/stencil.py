"""Stencil: concrete parametric pointwise stencil Function."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import Extent, _checked_bounds
from cosmic_foundry.theory.function import Function


@dataclass(frozen=True)
class Stencil(Function):
    """A pointwise stencil operator parametric over a kernel and radii.

    Function:
        domain   — (fields: Array of field arrays on Ω_h ⊆ ℤⁿ, extent: Extent)
        codomain — an array-valued field on Ω_h^int ⊆ Ω_h
        operator — pointwise application of fn over extent

    ``fn(fields, *index_meshgrids) -> scalar`` is the pointwise kernel;
    ``fields[i]`` accesses the i-th input field. ``radii`` gives the
    stencil half-widths per axis.

    ``execute(fields, extent=...)`` is provided automatically.
    """

    fn: Callable[..., Any]
    radii: tuple[int, ...]

    def execute(self, fields: Array[Any], *, extent: Extent) -> Any:
        _validate_halo_access(extent, self.radii, fields)
        return _make_jit_kernel(self.fn, extent)(*fields.elements)


@functools.lru_cache(maxsize=256)
def _make_jit_kernel(fn: Any, extent: Any) -> Callable[..., Any]:
    @jax.jit
    def apply(*jit_inputs: Any) -> Any:
        indices = _region_indices(extent)
        return fn(jit_inputs, *indices)

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
    fields: Array[Any],
) -> None:
    required = extent.expand(radii)
    for input_array in fields.elements:
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
