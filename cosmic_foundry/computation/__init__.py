"""Computation layer: distributed data containers and kernel execution."""

from __future__ import annotations

from cosmic_foundry.computation.array import Array, ComponentId, Placement, Record
from cosmic_foundry.computation.descriptor import (
    AccessPattern,
    Descriptor,
    Extent,
    Region,
)

__all__ = [
    "AccessPattern",
    "Array",
    "ComponentId",
    "Descriptor",
    "Extent",
    "Placement",
    "Record",
    "Region",
]
