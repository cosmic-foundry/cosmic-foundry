"""Stencil kernel execution engine and field reductions.

``execute_pointwise`` applies a stencil Function over a structured-grid
Region using JAX JIT compilation and optional vmap for block batching.

``GlobalSum`` sums field values over patch interiors with optional
MPI all-reduce.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.computation.array import Array, ComponentId
from cosmic_foundry.computation.descriptor import (
    AccessPattern,
    Extent,
    Region,
    _checked_bounds,
)
from cosmic_foundry.theory.function import Function


def execute_pointwise(
    fn: Any,
    region: Region,
    *field_arrays: Any,
) -> Any:
    """Apply a stencil fn over region with JAX JIT and input validation.

    ``fn`` must be hashable (for JIT caching) and expose:

    - ``_fn(*field_arrays, *index_meshgrids) -> scalar``
    - ``access_pattern: AccessPattern``

    When ``region.n_blocks > 1`` the kernel is lifted with ``jax.vmap``
    so ``_fn`` remains unaware of the batch dimension.
    """
    _validate_region_access(region, fn.access_pattern, field_arrays)
    return _make_jit_kernel(cast(Any, fn), region)(*field_arrays)


@functools.lru_cache(maxsize=256)
def _make_jit_kernel(fn: Any, region: Any) -> Callable[..., Any]:
    """Return a cached JIT-compiled kernel for *(fn, region)*.

    Both arguments must be hashable.  Frozen dataclasses satisfy this via
    their auto-generated ``__hash__``.  ``Region`` satisfies it as a frozen
    dataclass (with ``Extent.__hash__`` converting slice bounds to a tuple).

    Caching ensures repeated ``execute_pointwise`` calls with the same
    fn and Region reuse the same compiled XLA computation.
    """
    if region.n_blocks > 1:

        @jax.jit
        def apply_batched(*jit_inputs: Any) -> Any:
            indices = _region_indices(region)

            def single_block(*block_inputs: Any) -> Any:
                return fn._fn(*block_inputs, *indices)

            return jax.vmap(single_block)(*jit_inputs)

        return cast(Callable[..., Any], apply_batched)

    @jax.jit
    def apply(*jit_inputs: Any) -> Any:
        indices = _region_indices(region)
        return fn._fn(*jit_inputs, *indices)

    return cast(Callable[..., Any], apply)


def _region_indices(region: Region) -> tuple[jax.Array, ...]:
    axes = []
    for axis_slice in region.extent.slices:
        start, stop = _checked_bounds(axis_slice)
        axes.append(jnp.arange(start, stop))
    return tuple(jnp.meshgrid(*axes, indexing="ij"))


def _validate_region_access(
    region: Region,
    access_pattern: AccessPattern,
    inputs: tuple[Any, ...],
) -> None:
    required = region.extent.expand(access_pattern)
    batched = region.n_blocks > 1
    for input_array in inputs:
        if not hasattr(input_array, "shape"):
            msg = "Function inputs must expose a shape"
            raise TypeError(msg)
        shape = tuple(int(axis_size) for axis_size in input_array.shape)
        if batched:
            if shape[0] != region.n_blocks:
                msg = (
                    f"Input batch dimension {shape[0]} does not match "
                    f"Region.n_blocks={region.n_blocks}"
                )
                raise ValueError(msg)
            block_shape = shape[1:]
        else:
            block_shape = shape
        if len(block_shape) < required.ndim:
            msg = "Function input rank is smaller than the Region rank"
            raise ValueError(msg)
        for axis, axis_slice in enumerate(required.slices):
            start, stop = _checked_bounds(axis_slice)
            if start < 0 or stop > block_shape[axis]:
                msg = "Function Region plus access pattern exceeds input bounds"
                raise ValueError(msg)


@dataclass(frozen=True)
class GlobalSum(Function):
    """Sum field values over local interiors and optionally all-reduce them.

    Function:
        domain   — (mesh: Array[Patch], f_h : Ω_h^int → ℝ) — block mesh and
                   a discrete scalar field on interior grid points, intersected
                   with the given region
        codomain — ℝ (a real number; field evaluated at a single point)
        operator — (mesh, f_h, region) ↦ Σ_{x ∈ Ω_h^int ∩ region} f_h(x)

    Unweighted grid-point sum. To approximate ∫_Ω f dΩ, multiply by h^d
    where h is the grid spacing and d is the spatial dimension.

    Without *axis_name*, returns the rank-local sum. Supplying a JAX
    parallel-map axis name applies ``jax.lax.psum`` and returns the global
    sum inside that mapped context.

    Exact: Θ = ∅ — unweighted sum; no approximation introduced.
    """

    def execute(
        self,
        mesh: Any,
        field: Array[Any],
        region: Region,
        rank: int,
        *,
        axis_name: Hashable | None = None,
    ) -> jax.Array:
        local = jnp.asarray(0.0, dtype=jnp.float64)
        for i in range(len(mesh.elements)):
            cid = ComponentId(i)
            if cid not in mesh.placement.segments_for_rank(rank):
                continue
            block = mesh[cid]
            interior = block.index_extent
            overlap = _intersect_extents(interior, region.extent)
            if overlap is None:
                continue
            local = local + jnp.sum(field[cid][_payload_slices(interior, overlap)])

        if axis_name is None:
            return local
        return cast(jax.Array, jax.lax.psum(local, axis_name))


global_sum = GlobalSum()


def _payload_slices(parent: Extent, child: Extent) -> tuple[slice, ...]:
    return tuple(
        slice(
            child_slice.start - parent_slice.start,
            child_slice.stop - parent_slice.start,
        )
        for parent_slice, child_slice in zip(parent.slices, child.slices, strict=False)
    )


def _intersect_extents(a: Extent, b: Extent) -> Extent | None:
    if a.ndim != b.ndim:
        msg = "Cannot intersect Extents with different ndim"
        raise ValueError(msg)
    slices: list[slice] = []
    for sa, sb in zip(a.slices, b.slices, strict=False):
        start = max(sa.start, sb.start)
        stop = min(sa.stop, sb.stop)
        if start >= stop:
            return None
        slices.append(slice(start, stop))
    return Extent(tuple(slices))


__all__ = [
    "GlobalSum",
    "execute_pointwise",
    "global_sum",
]
