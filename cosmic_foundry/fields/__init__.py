"""Field, FieldSegment, Placement, and SegmentId data model.

The four concepts here correspond to the *storage* axis of the kernel
abstraction (ADR-0010):

- ``SegmentId``   — opaque identifier for one contiguous array segment.
- ``FieldSegment``— a payload array paired with the Extent over which it
                    is valid.  Carries no ownership information.
- ``Placement``   — maps each SegmentId to the process rank that owns it.
                    Carries no physical meaning or kernel-lowering logic.
- ``Field``       — a semantic label plus a tuple of FieldSegments plus a
                    Placement.  Does not own the iteration extent; Region
                    and Dispatch supply that.

Single-process execution is the degenerate case: one Field, one
FieldSegment, one Placement whose sole segment maps to rank 0.
Multi-process execution uses the same API with disjoint extents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NewType

import numpy as np

from cosmic_foundry.kernels import AccessPattern, Extent

SegmentId = NewType("SegmentId", int)


@dataclass(frozen=True)
class FieldSegment:
    """A payload array paired with the Extent over which it is valid.

    The payload is typically a ``jax.Array``.  ``FieldSegment`` does not
    own process/device information; that lives in ``Placement``.
    """

    segment_id: SegmentId
    payload: Any
    extent: Extent


class Placement:
    """Maps each ``SegmentId`` to the process rank that owns it.

    ``Placement`` carries no physical meaning and no kernel-lowering logic.
    It is the sole authoritative source for process/device ownership within
    a ``Field``.
    """

    def __init__(self, owners: Mapping[SegmentId, int]) -> None:
        if not owners:
            msg = "Placement must register at least one segment"
            raise ValueError(msg)
        for sid, rank in owners.items():
            if rank < 0:
                msg = f"Process rank must be non-negative; got rank={rank} for {sid!r}"
                raise ValueError(msg)
        self._owners: dict[SegmentId, int] = dict(owners)

    def owner(self, segment_id: SegmentId) -> int:
        """Return the rank that owns *segment_id*."""
        try:
            return self._owners[segment_id]
        except KeyError:
            msg = f"SegmentId {segment_id!r} is not registered in this Placement"
            raise KeyError(msg) from None

    def segments_for_rank(self, rank: int) -> frozenset[SegmentId]:
        """Return the set of SegmentIds owned by *rank*."""
        return frozenset(sid for sid, r in self._owners.items() if r == rank)

    def __repr__(self) -> str:
        return f"Placement({dict(self._owners)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Placement):
            return NotImplemented
        return self._owners == other._owners

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._owners.items())))


@dataclass(frozen=True)
class Field:
    """A named collection of FieldSegments with a process ownership map.

    ``Field`` owns the semantic label and the segment set.  It does not
    own iteration extent or process topology; those belong to ``Region``
    and ``Placement`` respectively.
    """

    name: str
    segments: tuple[FieldSegment, ...]
    placement: Placement

    def __post_init__(self) -> None:
        for seg in self.segments:
            try:
                self.placement.owner(seg.segment_id)
            except KeyError:
                msg = (
                    f"FieldSegment {seg.segment_id!r} is not registered "
                    f"in the Placement for Field {self.name!r}"
                )
                raise ValueError(msg) from None

    def segment(self, segment_id: SegmentId) -> FieldSegment:
        """Return the FieldSegment with the given *segment_id*."""
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        msg = f"SegmentId {segment_id!r} not found in Field {self.name!r}"
        raise KeyError(msg)

    def local_segments(self, rank: int) -> tuple[FieldSegment, ...]:
        """Return the FieldSegments owned by *rank* according to the Placement."""
        local_ids = self.placement.segments_for_rank(rank)
        return tuple(seg for seg in self.segments if seg.segment_id in local_ids)

    def covers(self, required_extent: Extent) -> bool:
        """Return True iff the union of all segment extents covers *required_extent*.

        Uses a boolean coverage mask; intended for validation, not hot paths.
        """
        shape = required_extent.shape
        origin = tuple(s.start for s in required_extent.slices)
        covered = np.zeros(shape, dtype=bool)
        for seg in self.segments:
            intersection = _intersect_extents(seg.extent, required_extent)
            if intersection is None:
                continue
            local_idx = tuple(
                slice(s.start - o, s.stop - o)
                for s, o in zip(intersection.slices, origin, strict=False)
            )
            covered[local_idx] = True
        return bool(covered.all())


def allocate_field(
    name: str,
    grid: Any,  # UniformGrid — imported lazily to keep mesh ↔ fields decoupled
    access_pattern: AccessPattern,
) -> Field:
    """Allocate a Field over a UniformGrid with halo-padded payloads.

    Each block in *grid* becomes one FieldSegment whose extent is the block's
    ``index_extent`` expanded by *access_pattern* and whose payload is a
    zero-initialised float64 JAX array of that expanded shape.  The returned
    Field's Placement mirrors *grid*'s rank assignment.
    """
    import jax.numpy as jnp

    segments: list[FieldSegment] = []
    owners: dict[SegmentId, int] = {}
    for block in grid.blocks:
        seg_id = SegmentId(int(block.block_id))
        halo_extent = block.index_extent.expand(access_pattern)
        payload = jnp.zeros(halo_extent.shape, dtype=jnp.float64)
        segments.append(
            FieldSegment(segment_id=seg_id, payload=payload, extent=halo_extent)
        )
        owners[seg_id] = grid.owner(block.block_id)

    return Field(name=name, segments=tuple(segments), placement=Placement(owners))


def _intersect_extents(a: Extent, b: Extent) -> Extent | None:
    """Return the intersection of two Extents, or None if the intersection is empty."""
    if a.ndim != b.ndim:
        msg = "Cannot intersect Extents with different ndim"
        raise ValueError(msg)
    slices: list[slice] = []
    for sa, sb in zip(a.slices, b.slices, strict=False):
        start = max(sa.start, sb.start)
        stop = min(sa.stop, sb.stop)
        if start >= stop:
            return None
        slices.append(slice(start, stop))
    return Extent(tuple(slices))


__all__ = [
    "Field",
    "FieldSegment",
    "Placement",
    "SegmentId",
    "allocate_field",
]
