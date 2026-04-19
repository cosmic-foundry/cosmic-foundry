"""Computation layer: distributed data containers and kernel execution."""

from __future__ import annotations

from cosmic_foundry.computation.array import Array, ComponentId, Placement, Record
from cosmic_foundry.computation.descriptor import (
    AccessPattern,
    Descriptor,
    Extent,
    Region,
)
from cosmic_foundry.computation.kernels import GlobalSum, global_sum

__all__ = [
    "AccessPattern",
    "Array",
    "ComponentId",
    "Descriptor",
    "Extent",
    "GlobalSum",
    "Placement",
    "Record",
    "Region",
    "global_sum",
]
