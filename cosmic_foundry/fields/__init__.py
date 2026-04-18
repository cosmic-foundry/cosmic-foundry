"""Field hierarchy, Placement, SegmentId, and FieldDiscretization.

- ``SegmentId``          — opaque identifier for one contiguous block segment.
- ``Placement``          — maps each SegmentId to the process rank that owns it.
                           Carries no physical meaning or kernel-lowering logic.
- ``Field``              — abstract base for all field parameterizations: f: D → ℝ.
- ``ContinuousField``    — Θ = ∅: f: D → ℝ represented by an analytic callable.
- ``DiscreteField``      — Θ = {h}: f_h: Ω_h → ℝ.  Leaf nodes (single block)
                           carry payload/extent/segment_id directly.  Composite
                           nodes (multi-block) carry segments and placement.
                           Both are the same kind of mathematical object — a
                           field on a discrete domain.
- ``FieldDiscretization``— map from ContinuousField × UniformGrid to DiscreteField
                           (spatial-domain implementation; concept is domain-general).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from cosmic_foundry.kernels import Descriptor, Extent, Map


@dataclass(frozen=True)
class SegmentId(Descriptor):
    """Opaque identifier for one contiguous block segment within a DiscreteField."""

    value: int

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value}


class Placement(Descriptor):
    """Maps each ``SegmentId`` to the process rank that owns it.

    ``Placement`` carries no physical meaning and no kernel-lowering logic.
    It is the sole authoritative source for process/device ownership within
    a composite ``DiscreteField``.
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

    def as_dict(self) -> dict[str, Any]:
        return {str(k): v for k, v in self._owners.items()}

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

    - ``ContinuousField``: D = any domain, Θ = ∅, stored as a callable.
    - ``DiscreteField``:   D = D_h ⊂ D,   Θ = {h}, stored as array segments.
    """

    name: str


@dataclass(frozen=True)
class ContinuousField(Field):
    """A continuous scalar field f: D → ℝ represented by an analytic callable.

    Θ = ∅ — exact representation; the callable is the field itself, not an
    approximation of it.  D may be any domain: physical space, thermodynamic
    state space, or otherwise.  Evaluated at a point in D by calling
    fn(*args) where each arg is a JAX array of coordinates along one axis
    of D.
    """

    name: str
    fn: Callable[..., Any]

    def evaluate(self, *args: Any) -> Any:
        """Evaluate the field at the given point in domain D."""
        import jax.numpy as jnp

        return jnp.asarray(self.fn(*args), dtype=jnp.float64)


@dataclass(frozen=True)
class DiscreteField(Field):
    """A discrete scalar field f_h: Ω_h → ℝ. Θ = {h}.

    Represents f_h on either a single block (leaf) or the full domain
    (composite).  Both are the same mathematical object — a field on a
    discrete domain — differing only in the extent of their domain D.

    Leaf (single block):
        ``segments = ()``.  ``segment_id``, ``payload``, and ``extent``
        are set.  ``interior_extent`` is optional and identifies the
        owned cells inside halo-padded storage.

    Composite (multi-block):
        ``segments`` is non-empty; each element is a leaf ``DiscreteField``.
        ``placement`` records which rank owns each segment.
        ``payload``, ``extent``, ``segment_id``, and ``interior_extent``
        are all ``None``.

    Approximation error is O(h^p) for smooth fields; p depends on the
    discretization scheme that produced this field.
    """

    name: str
    segments: tuple[DiscreteField, ...] = ()
    placement: Placement | None = None
    segment_id: SegmentId | None = None
    payload: Any | None = None
    extent: Extent | None = None
    interior_extent: Extent | None = None

    def __post_init__(self) -> None:
        if self.segments:
            # Composite node
            if self.placement is None:
                msg = f"composite DiscreteField {self.name!r} requires placement"
                raise ValueError(msg)
            if (
                self.payload is not None
                or self.extent is not None
                or self.segment_id is not None
            ):
                msg = (
                    f"composite DiscreteField {self.name!r} must not carry "
                    "payload, extent, or segment_id"
                )
                raise ValueError(msg)
            for seg in self.segments:
                try:
                    self.placement.owner(seg.segment_id)  # type: ignore[arg-type]
                except KeyError:
                    msg = (
                        f"DiscreteField segment {seg.segment_id!r} is not "
                        f"registered in the Placement for {self.name!r}"
                    )
                    raise ValueError(msg) from None
        else:
            # Leaf node
            if self.payload is None or self.extent is None or self.segment_id is None:
                msg = (
                    f"leaf DiscreteField {self.name!r} requires "
                    "payload, extent, and segment_id"
                )
                raise ValueError(msg)
            if self.placement is not None:
                msg = f"leaf DiscreteField {self.name!r} must not carry placement"
                raise ValueError(msg)
            if self.interior_extent is not None:
                intersection = _intersect_extents(self.extent, self.interior_extent)
                if intersection != self.interior_extent:
                    msg = "DiscreteField interior_extent must be contained in extent"
                    raise ValueError(msg)

    @property
    def is_leaf(self) -> bool:
        """True for a single-block (leaf) field."""
        return len(self.segments) == 0

    def segment(self, segment_id: SegmentId) -> DiscreteField:
        """Return the leaf DiscreteField with the given *segment_id*."""
        if self.is_leaf:
            if self.segment_id == segment_id:
                return self
            msg = f"SegmentId {segment_id!r} not found in DiscreteField {self.name!r}"
            raise KeyError(msg)
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        msg = f"SegmentId {segment_id!r} not found in DiscreteField {self.name!r}"
        raise KeyError(msg)

    def local_segments(self, rank: int) -> tuple[DiscreteField, ...]:
        """Return the leaf segments owned by *rank* according to the Placement."""
        if self.is_leaf:
            msg = "local_segments requires a composite DiscreteField with a Placement"
            raise ValueError(msg)
        local_ids = self.placement.segments_for_rank(rank)  # type: ignore[union-attr]
        return tuple(seg for seg in self.segments if seg.segment_id in local_ids)

    def covers(self, required_extent: Extent) -> bool:
        """Return True iff the union of segment extents covers *required_extent*.

        Uses a boolean coverage mask; intended for validation, not hot paths.
        """
        segs: tuple[DiscreteField, ...] = (self,) if self.is_leaf else self.segments
        shape = required_extent.shape
        origin = tuple(s.start for s in required_extent.slices)
        covered = np.zeros(shape, dtype=bool)
        for seg in segs:
            intersection = _intersect_extents(seg.extent, required_extent)  # type: ignore[arg-type]
            if intersection is None:
                continue
            local_idx = tuple(
                slice(s.start - o, s.stop - o)
                for s, o in zip(intersection.slices, origin, strict=False)
            )
            covered[local_idx] = True
        return bool(covered.all())


@dataclass(frozen=True)
class FieldDiscretization(Map):
    """Discretize a ContinuousField onto a discrete grid of points in its domain.

    The concept is domain-general: sampling f: D → ℝ onto D_h ⊂ D is the
    same operation whether D is physical space or thermodynamic state space.
    This implementation covers the spatial case (D = Ω ⊆ ℝⁿ, G = UniformGrid).

    Map:
        domain   — (f: ContinuousField on Ω ⊆ ℝⁿ, G = {(B_i, h)}) — a
                   continuous scalar field and a uniform grid partitioning Ω
                   into blocks B_i with grid spacing h; f.evaluate is called
                   with one JAX coordinate array per spatial axis,
                   broadcast-compatible with block shape
        codomain — f_h: DiscreteField on Ω_h — one leaf DiscreteField per
                   block with extent = block.index_extent and no ghost cells;
                   collected into a composite DiscreteField over the full grid
        operator — (f, G) ↦ f_h where f_h(x_i) = f(x_i) for x_i ∈ Ω_h^int

    Θ = {h}, p = 1 — piecewise-constant representation has L∞ error O(h)
    for smooth f; verified by MMS.
    """

    def execute(self, f: ContinuousField, grid: Any) -> DiscreteField:
        """Return a DiscreteField with payloads equal to f evaluated at cell centers."""
        import jax.numpy as jnp

        leaves: list[DiscreteField] = []
        owners: dict[SegmentId, int] = {}
        for block in grid.blocks:
            axes = [block.cell_centers(axis) for axis in range(block.ndim)]
            coords = jnp.meshgrid(*axes, indexing="ij")
            payload = jnp.asarray(f.evaluate(*coords), dtype=jnp.float64)
            seg_id = SegmentId(int(block.block_id))
            leaves.append(
                DiscreteField(
                    name=f.name,
                    segment_id=seg_id,
                    payload=payload,
                    extent=block.index_extent,
                )
            )
            owners[seg_id] = grid.owner(block.block_id)

        return DiscreteField(
            name=f.name,
            segments=tuple(leaves),
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
    "Placement",
    "SegmentId",
]
