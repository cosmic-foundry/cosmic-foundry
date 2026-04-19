"""Descriptor ABC and concrete descriptor types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class Descriptor(ABC):
    """Abstract base for all computation-configuring value objects.

    A Descriptor is an immutable object that specifies *how* computation is
    performed — coordinate extents, iteration regions, access patterns, field
    bindings, data placement. Descriptors are distinct from Records (which are
    *about* the simulation) and from Maps (which *perform* computation).

    Every Descriptor must be serializable to a plain dict.
    """

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this descriptor."""


@dataclass(frozen=True)
class AccessPattern(Descriptor):
    """Concrete descriptor specifying the locality contract for an Op.

    ``radii[i]`` is the ghost-cell depth required on axis ``i``.
    """

    radii: tuple[int, ...]

    def halo_width(self, axis: int) -> int:
        """Return the ghost-cell depth required on one axis."""
        return self.radii[axis]

    @classmethod
    def seven_point(cls) -> AccessPattern:
        """Return the 3-D seven-point nearest-neighbor access pattern."""
        return cls((1, 1, 1))

    @classmethod
    def symmetric(cls, order: int, ndim: int = 3) -> AccessPattern:
        """Return a symmetric access pattern for an even finite-difference order."""
        if order <= 0 or order % 2 != 0:
            msg = "symmetric order must be a positive even integer"
            raise ValueError(msg)
        if ndim <= 0:
            msg = "ndim must be positive"
            raise ValueError(msg)
        return cls((order // 2,) * ndim)

    def as_dict(self) -> dict[str, Any]:
        return {"type": "AccessPattern", "radii": list(self.radii)}


@dataclass(frozen=True)
class Extent(Descriptor):
    """Half-open integer index extent."""

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

    def expand(self, access_pattern: AccessPattern) -> Extent:
        """Return the extent plus the halo required by an access pattern."""
        expanded = []
        for axis, axis_slice in enumerate(self.slices):
            start, stop = _checked_bounds(axis_slice)
            halo = access_pattern.halo_width(axis)
            expanded.append(slice(start - halo, stop + halo))
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


__all__ = [
    "AccessPattern",
    "Descriptor",
    "Extent",
    "Region",
]
