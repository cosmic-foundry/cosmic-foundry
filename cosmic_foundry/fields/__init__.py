"""Field hierarchy, FieldSegment, Placement, SegmentId, and FieldDiscretization.

- ``SegmentId``          — opaque identifier for one contiguous array segment.
- ``FieldSegment``       — a payload array paired with the Extent over which it
                           is valid.  Carries no ownership information.
- ``Placement``          — maps each SegmentId to the process rank that owns it.
                           Carries no physical meaning or kernel-lowering logic.
- ``Field``              — abstract base for all field parameterizations: f: D → ℝ.
- ``ContinuousField``    — Θ = ∅: f: Ω → ℝ represented by an analytic callable.
- ``DiscreteField``      — Θ = {h}: f_h: Ω_h → ℝ stored as per-block array segments.
- ``FieldDiscretization``— map from ContinuousField × UniformGrid to DiscreteField.

Single-process execution is the degenerate case: one DiscreteField, one
FieldSegment, one Placement whose sole segment maps to rank 0.
Multi-process execution uses the same API with disjoint extents.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NewType

import numpy as np

from cosmic_foundry.kernels import Extent

SegmentId = NewType("SegmentId", int)


@dataclass(frozen=True)
class FieldSegment:
    """A payload array paired with the Extent over which it is valid.

    The payload is typically a ``jax.Array``.  ``FieldSegment`` does not
    own process/device information; that lives in ``Placement``.
    ``interior_extent`` identifies the owned cells inside halo-padded
    storage; when omitted, the whole segment extent is treated as interior.
    """

    segment_id: SegmentId
    payload: Any
    extent: Extent
    interior_extent: Extent | None = None

    def __post_init__(self) -> None:
        if self.interior_extent is None:
            return
        intersection = _intersect_extents(self.extent, self.interior_extent)
        if intersection != self.interior_extent:
            msg = "FieldSegment interior_extent must be contained in extent"
            raise ValueError(msg)


class Placement:
    """Maps each ``SegmentId`` to the process rank that owns it.

    ``Placement`` carries no physical meaning and no kernel-lowering logic.
    It is the sole authoritative source for process/device ownership within
    a ``DiscreteField``.
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


class Field(ABC):
    """Abstract base for all field parameterizations: f: D → ℝ.

    A field assigns a value to every point in its domain D. Concrete
    subclasses differ in how D is represented and how f is stored:

    - ``ContinuousField``: D = Ω ⊆ ℝⁿ, Θ = ∅, stored as a callable.
    - ``DiscreteField``:   D = Ω_h ⊂ Ω,  Θ = {h}, stored as array segments.
    """

    name: str


@dataclass(frozen=True)
class ContinuousField(Field):
    """A continuous scalar field f: Ω → ℝ represented by an analytic callable.

    Θ = ∅ — exact representation; the callable is the field itself, not an
    approximation of it.  Evaluated at arbitrary spatial coordinates by
    calling fn(*coords) where each coord is a JAX array of positions along
    one spatial axis.
    """

    name: str
    fn: Callable[..., Any]

    def evaluate(self, *coords: Any) -> Any:
        """Evaluate the field at the given spatial coordinates."""
        import jax.numpy as jnp

        return jnp.asarray(self.fn(*coords), dtype=jnp.float64)


@dataclass(frozen=True)
class DiscreteField(Field):
    """A discrete scalar field f_h: Ω_h → ℝ stored as per-block array segments.

    Θ = {h} — the discrete representation approximates the underlying continuous
    field to O(h) in the L∞ norm for smooth fields under piecewise-constant
    interpolation.  Produced by ``FieldDiscretization``; consumed by halo-fill,
    kernel dispatch, and diagnostic reduction maps.
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
                    f"in the Placement for DiscreteField {self.name!r}"
                )
                raise ValueError(msg) from None

    def segment(self, segment_id: SegmentId) -> FieldSegment:
        """Return the FieldSegment with the given *segment_id*."""
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        msg = f"SegmentId {segment_id!r} not found in DiscreteField {self.name!r}"
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


@dataclass(frozen=True)
class FieldDiscretization:
    """Discretize a ContinuousField onto interior cell centers of a uniform grid.

    Map:
        domain   — (f: ContinuousField, G = {(B_i, h)}) — a continuous scalar
                   field and a uniform grid partitioning Ω into blocks B_i with
                   grid spacing h; f.evaluate is called with one JAX coordinate
                   array per spatial axis, broadcast-compatible with block shape
        codomain — f_h: DiscreteField on Ω_h^int(B_i) — one FieldSegment per
                   block with extent = block.index_extent and no ghost cells
        operator — (f, G) ↦ f_h where f_h(x_i) = f(x_i) for x_i ∈ Ω_h^int

    Θ = {h}, p = 1 — piecewise-constant representation has L∞ error O(h)
    for smooth f; verified by MMS.
    """

    def execute(self, f: ContinuousField, grid: Any) -> DiscreteField:
        """Return a DiscreteField with payloads equal to f evaluated at cell centers."""
        import jax.numpy as jnp

        segments: list[FieldSegment] = []
        owners: dict[SegmentId, int] = {}
        for block in grid.blocks:
            axes = [block.cell_centers(axis) for axis in range(block.ndim)]
            coords = jnp.meshgrid(*axes, indexing="ij")
            payload = jnp.asarray(f.evaluate(*coords), dtype=jnp.float64)
            seg_id = SegmentId(int(block.block_id))
            segments.append(
                FieldSegment(
                    segment_id=seg_id,
                    payload=payload,
                    extent=block.index_extent,
                )
            )
            owners[seg_id] = grid.owner(block.block_id)

        return DiscreteField(
            name=f.name,
            segments=tuple(segments),
            placement=Placement(owners),
        )


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
    "ContinuousField",
    "DiscreteField",
    "Field",
    "FieldDiscretization",
    "FieldSegment",
    "Placement",
    "SegmentId",
]
