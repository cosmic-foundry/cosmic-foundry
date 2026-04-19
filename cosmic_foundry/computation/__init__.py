"""Computation layer: distributed data containers and kernel execution."""

from __future__ import annotations

from cosmic_foundry.computation.array import Array, ComponentId, Placement
from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.computation.reductions import GlobalSum, global_sum
from cosmic_foundry.computation.stencil import Stencil

__all__ = [
    "Array",
    "ComponentId",
    "Extent",
    "GlobalSum",
    "Placement",
    "Stencil",
    "global_sum",
]
