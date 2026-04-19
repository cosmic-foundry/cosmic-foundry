"""Field sampling onto a mesh.

discretize() is a placeholder free function. Once Field becomes the
simulation state carrier and the geometry/ layer (Domain, ManifoldWithBoundary)
is complete, this will be replaced by a proper Function subclass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from cosmic_foundry.computation.array import Array


def discretize(fn: Callable[..., Any], mesh: Array) -> Array:
    """Sample fn at each patch's cell-center node positions.

    fn receives one JAX coordinate array per axis (via meshgrid) and must
    return a JAX array of the same shape.  The returned Array has one element
    per patch, each with shape equal to patch.index_extent.shape.
    """
    elements: list[Any] = []
    for patch in mesh.elements:
        axes = [patch.node_positions(axis) for axis in range(patch.ndim)]
        coords = jnp.meshgrid(*axes, indexing="ij")
        elements.append(jnp.asarray(fn(*coords), dtype=jnp.float64))
    return Array(elements=tuple(elements))


__all__ = ["discretize"]
