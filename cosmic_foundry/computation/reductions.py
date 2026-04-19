"""Field reductions.

``Reduction`` folds field values over patch interiors using a caller-supplied
aggregation function and identity element, with optional all-reduce.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import (
    Extent,
    intersect_extents,
    payload_slices,
)
from cosmic_foundry.theory.function import Function


@dataclass(frozen=True)
class Reduction(Function):
    """A field reduction parametric over an aggregation operator and identity.

    Function:
        domain   — (mesh: Array[Patch], f_h : Ω_h^int → ℝ, extent: Extent)
        codomain — ℝ
        operator — (mesh, f_h, extent) ↦
                       fold_{x ∈ Ω_h^int ∩ extent} f_h(x)
                   where fold is defined by (operator, identity)

    Θ = ∅ — the reduction is exact given the operator and identity.

    Without *axis_name*, returns the local reduced value. Supplying a JAX
    parallel-map axis name applies ``jax.lax.psum`` after the local fold
    (valid when operator is addition).
    """

    operator: Callable[..., jax.Array]
    identity: float

    def execute(
        self,
        mesh: Any,
        field: Array[Any],
        extent: Extent,
        *,
        axis_name: Hashable | None = None,
    ) -> jax.Array:
        result = jnp.asarray(self.identity, dtype=jnp.float64)
        for i in range(len(mesh.elements)):
            interior = mesh[i].index_extent
            overlap = intersect_extents(interior, extent)
            if overlap is None:
                continue
            result = result + self.operator(field[i][payload_slices(interior, overlap)])

        if axis_name is None:
            return result
        return cast(jax.Array, jax.lax.psum(result, axis_name))


global_sum = Reduction(operator=jnp.sum, identity=0.0)

__all__ = ["Reduction", "global_sum"]
