"""Extent: finite rectangular subset of ℤⁿ."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cosmic_foundry.theory.indexed_set import IndexedSet


@dataclass(frozen=True)
class Extent(IndexedSet):
    """Half-open integer index extent — a finite rectangular subset of ℤⁿ."""

    slices: tuple[slice, ...]

    def __hash__(self) -> int:
        # slice objects are not hashable; derive the hash from their bounds.
        return hash(tuple((s.start, s.stop) for s in self.slices))

    @classmethod
    def from_shape(cls, shape: Sequence[int]) -> Extent:
        """Create an extent covering a full array shape."""
        return cls(tuple(slice(0, int(n)) for n in shape))

    @property
    def ndim(self) -> int:
        """Number of axes in the extent."""
        return len(self.slices)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape implied by the extent."""
        return tuple(_slice_length(axis_slice) for axis_slice in self.slices)

    def intersect(self, other: IndexedSet) -> Extent | None:
        """Return S ∩ T as an Extent, or None if the extents are disjoint."""
        if not isinstance(other, Extent):
            raise TypeError(f"Cannot intersect Extent with {type(other).__name__}")
        if self.ndim != other.ndim:
            raise ValueError("Cannot intersect Extents with different ndim")
        slices: list[slice] = []
        for sa, sb in zip(self.slices, other.slices, strict=False):
            start = max(sa.start, sb.start)
            stop = min(sa.stop, sb.stop)
            if start >= stop:
                return None
            slices.append(slice(start, stop))
        return Extent(tuple(slices))


def _checked_bounds(axis_slice: slice) -> tuple[int, int]:
    if axis_slice.start is None or axis_slice.stop is None:
        msg = "Extent slices must have explicit start and stop"
        raise ValueError(msg)
    if axis_slice.step not in (None, 1):
        msg = "Extent slices do not support non-unit steps"
        raise ValueError(msg)
    return int(axis_slice.start), int(axis_slice.stop)


def _slice_length(axis_slice: slice) -> int:
    start, stop = _checked_bounds(axis_slice)
    length = stop - start
    if length < 0:
        msg = "Extent slices must be half-open with stop >= start"
        raise ValueError(msg)
    return length


__all__ = ["Extent"]
