"""Computation layer: distributed data containers and kernel execution."""

from __future__ import annotations

from cosmic_foundry.computation.array import Array, ComponentId, Placement
from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.computation.reductions import GlobalSum, global_sum

__all__ = [
    "Array",
    "ComponentId",
    "Extent",
    "GlobalSum",
    "Placement",
    "global_sum",
]
