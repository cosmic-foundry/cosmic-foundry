"""Uniform structured mesh."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from cosmic_foundry.descriptor import AccessPattern, Extent
from cosmic_foundry.domain import Domain
from cosmic_foundry.map import Map
from cosmic_foundry.record import ComponentId, Placement


@dataclass(frozen=True)
class Block(Domain):
    """One contiguous patch of uniformly-spaced cells — a spatial sub-domain.

    Owns topology and coordinate metadata only; array payloads live in Field.
    """

    block_id: ComponentId
    index_extent: Extent
    origin: tuple[float, ...]  # physical coord of first cell center
    cell_spacing: tuple[float, ...]  # h_i along each axis

    @property
    def ndim(self) -> int:
        return self.index_extent.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.index_extent.shape

    def cell_centers(self, axis: int) -> Any:
        """1-D coordinate array of cell centers along *axis*."""
        import jax.numpy as jnp

        return (
            self.origin[axis] + jnp.arange(self.shape[axis]) * self.cell_spacing[axis]
        )


@dataclass(frozen=True)
class UniformGrid(Domain):
    """Ω_h — a continuous domain Ω partitioned into a structured block grid.

    Produced by :data:`partition_domain`.  Blocks are enumerated in C order
    (last axis varies fastest) and distributed across ranks round-robin by
    flat index.
    """

    blocks: tuple[Block, ...]
    rank_map: tuple[int, ...]  # rank_map[block_id.value] → owning rank

    @property
    def ndim(self) -> int:
        return self.blocks[0].ndim

    def block(self, block_id: ComponentId) -> Block:
        return self.blocks[block_id.value]

    def owner(self, block_id: ComponentId) -> int:
        return self.rank_map[block_id.value]

    def blocks_for_rank(self, rank: int) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if self.rank_map[b.block_id.value] == rank)

    def fill_halo(
        self,
        field: DistributedField,
        access_pattern: AccessPattern,
        rank: int,
    ) -> DistributedField:
        """Return a new DistributedField with same-rank ghost cells filled.

        Copies ghost-cell values from the interior of neighboring segments
        owned by the same rank.  Multi-rank halo exchange is not implemented;
        a NotImplementedError is raised if a ghost cell's source lives on a
        different rank.
        """
        local_ids: set[ComponentId] = {
            seg.segment_id for seg in field.local_segments(rank)
        }
        updated: dict[ComponentId, Any] = {}
        for target in field.local_segments(rank):
            updated[target.segment_id] = _fill_segment_halo(
                target=target,
                field=field,
                rank=rank,
                local_ids=local_ids,
                access_pattern=access_pattern,
            )
        new_segments = tuple(
            FieldSegment(
                name=seg.name,
                segment_id=seg.segment_id,
                payload=updated.get(seg.segment_id, seg.payload),
                extent=seg.extent,
                interior_extent=seg.interior_extent,
            )
            for seg in field.segments
        )
        return DistributedField(
            name=field.name,
            segments=new_segments,
            placement=field.placement,
        )


@dataclass(frozen=True)
class PartitionDomain(Map):
    """Partition a continuous domain into a discrete block grid.

    Map:
        domain   — (Ω = ∏ᵢ [oᵢ, oᵢ+Lᵢ], n_cells ∈ ℤⁿ,
                   blocks_per_axis ∈ ℤⁿ, n_ranks ∈ ℤ) — a continuous
                   domain specification and discretization parameters
        codomain — UniformGrid (Ω_h): a partition of Ω into
                   ∏ blocks_per_axis blocks, each covering
                   n_cells/blocks_per_axis interior cells with spacing
                   h = L/n_cells, assigned to ranks round-robin
        operator — (Ω, h, blocks) ↦ {(B_i, Ω_h^int(B_i), rank_i)}_i

    Θ = {h} — the discrete grid approximates the continuous domain;
    h = domain_size / n_cells along each axis.
    """

    def execute(
        self,
        *,
        domain_origin: tuple[float, ...],
        domain_size: tuple[float, ...],
        n_cells: tuple[int, ...],
        blocks_per_axis: tuple[int, ...],
        n_ranks: int,
    ) -> UniformGrid:
        ndim = len(n_cells)
        if not (len(domain_origin) == len(domain_size) == len(blocks_per_axis) == ndim):
            raise ValueError(
                "domain_origin, domain_size, n_cells, and blocks_per_axis "
                "must all have the same length"
            )
        for i in range(ndim):
            if n_cells[i] % blocks_per_axis[i] != 0:
                raise ValueError(
                    f"n_cells[{i}]={n_cells[i]} is not divisible by "
                    f"blocks_per_axis[{i}]={blocks_per_axis[i]}"
                )

        h = tuple(domain_size[i] / n_cells[i] for i in range(ndim))
        cpb = tuple(n_cells[i] // blocks_per_axis[i] for i in range(ndim))

        blocks: list[Block] = []
        rank_map: list[int] = []

        for flat_id, multi_idx in enumerate(
            itertools.product(*(range(blocks_per_axis[i]) for i in range(ndim)))
        ):
            starts = tuple(multi_idx[i] * cpb[i] for i in range(ndim))
            stops = tuple(starts[i] + cpb[i] for i in range(ndim))
            index_extent = Extent(
                tuple(slice(starts[i], stops[i]) for i in range(ndim))
            )
            origin = tuple(
                domain_origin[i] + (starts[i] + 0.5) * h[i] for i in range(ndim)
            )
            blocks.append(
                Block(
                    block_id=ComponentId(flat_id),
                    index_extent=index_extent,
                    origin=origin,
                    cell_spacing=h,
                )
            )
            rank_map.append(flat_id % n_ranks)

        return UniformGrid(blocks=tuple(blocks), rank_map=tuple(rank_map))


partition_domain = PartitionDomain()


@dataclass(frozen=True)
class FieldSegment:
    """A discrete scalar field on one spatial block: f_h: B_h → ℝ.

    Carries the array payload together with the spatial metadata that
    locates it within the global domain.  ``interior_extent`` is set
    only when ghost cells are present; without it the full ``extent``
    is the interior.
    """

    name: str
    segment_id: ComponentId
    payload: Any
    extent: Extent
    interior_extent: Extent | None = None

    def __post_init__(self) -> None:
        if self.interior_extent is not None:
            intersection = _intersect_extents(self.extent, self.interior_extent)
            if intersection != self.interior_extent:
                msg = "FieldSegment interior_extent must be contained in extent"
                raise ValueError(msg)


@dataclass(frozen=True)
class DistributedField:
    """A discrete scalar field over the full domain, partitioned into blocks.

    Each segment is a ``FieldSegment`` owned by exactly one rank according
    to ``placement``.  ``DistributedField`` is the spatial-level counterpart
    of a ``DiscreteField``; it carries ownership and topology metadata that
    the pure ``DiscreteField`` deliberately omits.
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
                    f"DistributedField segment {seg.segment_id!r} is not "
                    f"registered in the Placement for {self.name!r}"
                )
                raise ValueError(msg) from None

    def segment(self, segment_id: ComponentId) -> FieldSegment:
        """Return the segment with the given *segment_id*."""
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        msg = f"ComponentId {segment_id!r} not found in DistributedField {self.name!r}"
        raise KeyError(msg)

    def local_segments(self, rank: int) -> tuple[FieldSegment, ...]:
        """Return the segments owned by *rank* according to the placement."""
        local_ids = self.placement.segments_for_rank(rank)
        return tuple(seg for seg in self.segments if seg.segment_id in local_ids)

    def covers(self, required_extent: Extent) -> bool:
        """Return True iff the union of segment extents covers *required_extent*."""
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
class FieldDiscretization(Map):
    """Discretize a ContinuousField onto a discrete grid of points in its domain.

    The concept is domain-general: sampling f: D → ℝ onto D_h ⊂ D is the
    same operation whether D is physical space or thermodynamic state space.
    This implementation covers the spatial case (D = Ω ⊆ ℝⁿ, G = UniformGrid).

    Map:
        domain   — (f: ContinuousField on Ω ⊆ ℝⁿ, G = {(B_i, h)}) — a
                   continuous scalar field and a uniform grid partitioning Ω
                   into blocks B_i with grid spacing h; f.sample is called
                   with one JAX coordinate array per spatial axis,
                   broadcast-compatible with block shape
        codomain — f_h: DistributedField on Ω_h — one FieldSegment per block
                   with extent = block.index_extent and no ghost cells;
                   collected into a DistributedField over the full grid
        operator — (f, G) ↦ f_h where f_h(x_i) = f(x_i) for x_i ∈ Ω_h^int

    Θ = {h}, p = 1 — piecewise-constant representation has L∞ error O(h)
    for smooth f; verified by MMS.
    """

    def execute(self, f: Any, grid: Any) -> DistributedField:
        """Return a DistributedField with payloads equal to f at cell centers."""
        import jax.numpy as jnp

        leaves: list[FieldSegment] = []
        owners: dict[ComponentId, int] = {}
        for block in grid.blocks:
            axes = [block.cell_centers(axis) for axis in range(block.ndim)]
            coords = jnp.meshgrid(*axes, indexing="ij")
            payload = f.sample(*coords).payload
            leaves.append(
                FieldSegment(
                    name=f.name,
                    segment_id=block.block_id,
                    payload=payload,
                    extent=block.index_extent,
                )
            )
            owners[block.block_id] = grid.owner(block.block_id)

        return DistributedField(
            name=f.name,
            segments=tuple(leaves),
            placement=Placement(owners),
        )


field_discretization = FieldDiscretization()


def _fill_segment_halo(
    *,
    target: FieldSegment,
    field: DistributedField,
    rank: int,
    local_ids: set[ComponentId],
    access_pattern: AccessPattern,
) -> Any:
    interior = _segment_interior(target, access_pattern)
    payload = target.payload
    for halo_piece in _subtract_extent(target.extent, interior):
        candidates = _source_candidates(
            field=field,
            target=target,
            halo_piece=halo_piece,
            access_pattern=access_pattern,
        )
        local_candidates = [
            (src, overlap) for src, overlap in candidates if src.segment_id in local_ids
        ]
        if len(local_candidates) > 1:
            msg = "Multiple same-rank source segments overlap one halo region"
            raise ValueError(msg)
        if len(local_candidates) == 0:
            if candidates:
                msg = (
                    f"fill_halo cannot fill a halo from rank {rank}; "
                    "multi-rank halo exchange is not implemented"
                )
                raise NotImplementedError(msg)
            continue
        src, overlap = local_candidates[0]
        payload = payload.at[_payload_slices(target.extent, overlap)].set(
            src.payload[_payload_slices(src.extent, overlap)]
        )
    return payload


def _source_candidates(
    *,
    field: DistributedField,
    target: FieldSegment,
    halo_piece: Extent,
    access_pattern: AccessPattern,
) -> list[tuple[FieldSegment, Extent]]:
    candidates: list[tuple[FieldSegment, Extent]] = []
    for src in field.segments:
        if src.segment_id == target.segment_id:
            continue
        src_interior = _segment_interior(src, access_pattern)
        overlap = _intersect_extents(src_interior, halo_piece)
        if overlap is not None:
            candidates.append((src, overlap))
    return candidates


def _segment_interior(segment: FieldSegment, access_pattern: AccessPattern) -> Extent:
    if segment.interior_extent is not None:
        return segment.interior_extent
    return _shrink_extent(segment.extent, access_pattern)


def _shrink_extent(extent: Extent, access_pattern: AccessPattern) -> Extent:
    slices: list[slice] = []
    for axis, axis_slice in enumerate(extent.slices):
        halo = access_pattern.halo_width(axis)
        start = axis_slice.start + halo
        stop = axis_slice.stop - halo
        if start > stop:
            msg = "Cannot shrink an extent by a halo wider than the extent"
            raise ValueError(msg)
        slices.append(slice(start, stop))
    return Extent(tuple(slices))


def _subtract_extent(extent: Extent, removed: Extent) -> tuple[Extent, ...]:
    """Return pieces of *extent* that lie outside *removed*."""
    overlap = _intersect_extents(extent, removed)
    if overlap is None:
        return (extent,)
    axis_parts: list[list[tuple[slice, bool]]] = []
    for axis_extent, axis_overlap in zip(extent.slices, overlap.slices, strict=False):
        parts: list[tuple[slice, bool]] = []
        if axis_extent.start < axis_overlap.start:
            parts.append((slice(axis_extent.start, axis_overlap.start), False))
        parts.append((axis_overlap, True))
        if axis_overlap.stop < axis_extent.stop:
            parts.append((slice(axis_overlap.stop, axis_extent.stop), False))
        axis_parts.append(parts)
    pieces: list[Extent] = []
    for combo in itertools.product(*axis_parts):
        slices = tuple(part for part, _ in combo)
        if all(inside for _, inside in combo):
            continue
        if all(s.start < s.stop for s in slices):
            pieces.append(Extent(slices))
    return tuple(pieces)


def _payload_slices(parent: Extent, child: Extent) -> tuple[slice, ...]:
    return tuple(
        slice(c.start - p.start, c.stop - p.start)
        for p, c in zip(parent.slices, child.slices, strict=False)
    )


def _intersect_extents(a: Extent, b: Extent) -> Extent | None:
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
    "Block",
    "DistributedField",
    "FieldDiscretization",
    "FieldSegment",
    "PartitionDomain",
    "UniformGrid",
    "field_discretization",
    "partition_domain",
]
