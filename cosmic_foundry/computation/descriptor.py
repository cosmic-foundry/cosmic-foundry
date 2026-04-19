"""Descriptor ABC and concrete descriptor types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cosmic_foundry.theory.indexed_set import IndexedSet


class Descriptor(ABC):
    """Abstract base for all computation-configuring value objects.

    A Descriptor is an immutable object that specifies *how* computation is
    performed — coordinate extents, iteration regions, access patterns, field
    bindings, data placement. Descriptors are distinct from Maps (which *perform*
    computation).

    Every Descriptor must be serializable to a plain dict.
    """

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this descriptor."""


@dataclass(frozen=True)
class Extent(Descriptor, IndexedSet):
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

    def expand(self, radii: tuple[int, ...]) -> Extent:
        """Return the extent grown by *radii* ghost cells on each axis."""
        expanded = []
        for axis, axis_slice in enumerate(self.slices):
            start, stop = _checked_bounds(axis_slice)
            expanded.append(slice(start - radii[axis], stop + radii[axis]))
        return Extent(tuple(expanded))

    def as_dict(self) -> dict[str, Any]:
        return {"slices": [(s.start, s.stop) for s in self.slices]}


@dataclass(frozen=True)
class Region(Descriptor):
    """Iteration coordinates for a Function execution.

    ``n_blocks`` signals a batched region: inputs are expected to carry a
    leading batch axis of that size, and ``execute_pointwise`` lowers the
    kernel with ``jax.vmap`` so the Function remains unaware of the batch dimension.
    """

    extent: Extent
    n_blocks: int = 1

    def __post_init__(self) -> None:
        if self.n_blocks < 1:
            msg = "Region.n_blocks must be a positive integer"
            raise ValueError(msg)

    def as_dict(self) -> dict[str, Any]:
        return {"extent": self.extent.as_dict(), "n_blocks": self.n_blocks}


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


def intersect_extents(a: Extent, b: Extent) -> Extent | None:
    """Return the intersection of two Extents, or None if they do not overlap."""
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


def payload_slices(parent: Extent, child: Extent) -> tuple[slice, ...]:
    """Return slices that index *child* within an array sized for *parent*."""
    return tuple(
        slice(
            child_slice.start - parent_slice.start,
            child_slice.stop - parent_slice.start,
        )
        for parent_slice, child_slice in zip(parent.slices, child.slices, strict=False)
    )


__all__ = [
    "Descriptor",
    "Extent",
    "Region",
    "intersect_extents",
    "payload_slices",
]
