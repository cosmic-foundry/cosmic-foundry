"""Overlap-based array copy.

``fill_by_overlap`` copies data from one array into another wherever their
index extents intersect, with no assumptions about array shapes beyond what
the extents imply.
"""

from __future__ import annotations

from typing import Any

from cosmic_foundry.computation.descriptor import (
    Extent,
    intersect_extents,
    payload_slices,
)


def fill_by_overlap(
    source_extent: Extent,
    source_array: Any,
    target_extent: Extent,
    target_array: Any,
) -> Any:
    """Copy from source_array into target_array on the intersection of their extents.

    Returns the updated target_array, or the original if the extents do not intersect.
    """
    overlap = intersect_extents(source_extent, target_extent)
    if overlap is None:
        return target_array
    return target_array.at[payload_slices(target_extent, overlap)].set(
        source_array[payload_slices(source_extent, overlap)]
    )


__all__ = ["fill_by_overlap"]
