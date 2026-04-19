"""Computation layer: distributed data containers and kernel execution."""

from __future__ import annotations

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.computation.reductions import GlobalSum, global_sum
from cosmic_foundry.computation.stencil import Stencil

__all__ = [
    "Array",
    "Extent",
    "GlobalSum",
    "Stencil",
    "global_sum",
]
