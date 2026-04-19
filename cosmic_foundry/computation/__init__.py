"""Computation layer: distributed data containers and kernel execution."""

from __future__ import annotations

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.computation.field import discretize
from cosmic_foundry.computation.reductions import Reduction, global_sum
from cosmic_foundry.computation.stencil import Stencil

__all__ = [
    "Array",
    "Extent",
    "Reduction",
    "Stencil",
    "discretize",
    "global_sum",
]
