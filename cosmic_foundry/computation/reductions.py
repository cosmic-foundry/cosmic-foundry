"""Field reductions.

``GlobalSum`` sums field values over patch interiors with optional
MPI all-reduce.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.computation.array import Array, ComponentId
from cosmic_foundry.computation.descriptor import (
    Region,
    intersect_extents,
    payload_slices,
)
from cosmic_foundry.theory.function import Function


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
            overlap = intersect_extents(interior, region.extent)
            if overlap is None:
                continue
            local = local + jnp.sum(field[cid][payload_slices(interior, overlap)])

        if axis_name is None:
            return local
        return cast(jax.Array, jax.lax.psum(local, axis_name))


global_sum = GlobalSum()

__all__ = ["GlobalSum", "global_sum"]
