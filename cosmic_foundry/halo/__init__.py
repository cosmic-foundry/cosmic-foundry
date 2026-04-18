"""Halo-fill fence and local halo-copy policy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from cosmic_foundry.fields import DiscreteField
from cosmic_foundry.kernels import (
    AccessPattern,
    ComponentId,
    Descriptor,
    Extent,
    Map,
    Region,
)


@dataclass(frozen=True)
class HaloFillFence(Descriptor):
    """Communication intent for one field before a Map execution."""

    field: DiscreteField
    region: Region
    access_pattern: AccessPattern

    def as_dict(self) -> dict[str, Any]:
        return {
            "region": self.region.as_dict(),
            "access_pattern": self.access_pattern.as_dict(),
        }


@dataclass(frozen=True)
class HaloFillPolicy(Map):
    """Fill same-rank ghost cells by copying from neighboring segments.

    Map:
        domain   — f_h : Ω_h^int(B_i) → ℝⁿ, a rank-n discrete field
                   defined on the interior cells of each block B_i
        codomain — f_h : Ω_h^ext(B_i) → ℝⁿ, the same field extended
                   to interior ∪ ghost cells (Ω_h^ext = Ω_h^int ∪ Ω_h^ghost)
        operator — f̃(x) = f(x) for x ∈ Ω_h^int;
                   f̃(x) = f_adj(x) for x ∈ Ω_h^ghost,
                   where f_adj is the interior value of the unique
                   adjacent segment whose interior contains x

    Exact: Θ = ∅ — domain extension by direct copy; no approximation.
    """

    def execute(self, fence: HaloFillFence, rank: int) -> DiscreteField:
        """Return a new Field with ghost cells filled for *rank*."""
        required = fence.region.extent.expand(fence.access_pattern)
        if not fence.field.covers(required):
            msg = "Field does not cover the Region plus halo required by the fence"
            raise ValueError(msg)

        local_ids: set[ComponentId] = {
            seg.segment_id
            for seg in fence.field.local_segments(rank)
            if seg.segment_id is not None  # always true for leaves; narrows type
        }
        updated_payloads: dict[ComponentId, Any] = {}
        for target in fence.field.local_segments(rank):
            assert target.segment_id is not None  # target is a leaf
            target_interior = _segment_interior(target, fence.access_pattern)
            if _intersect_extents(target_interior, fence.region.extent) is None:
                continue
            updated_payloads[target.segment_id] = _fill_segment_halo(
                target=target,
                field=fence.field,
                rank=rank,
                required=required,
                interior=fence.region.extent,
                access_pattern=fence.access_pattern,
                local_ids=local_ids,
            )

        new_segments_list: list[DiscreteField] = []
        for seg in fence.field.segments:
            assert seg.segment_id is not None and seg.extent is not None  # leaf
            new_segments_list.append(
                DiscreteField(
                    name=seg.name,
                    segment_id=seg.segment_id,
                    payload=updated_payloads.get(seg.segment_id, seg.payload),
                    extent=seg.extent,
                    interior_extent=seg.interior_extent,
                )
            )
        new_segments = tuple(new_segments_list)
        return DiscreteField(
            name=fence.field.name,
            segments=new_segments,
            placement=fence.field.placement,
        )


def _fill_segment_halo(
    *,
    target: DiscreteField,
    field: DiscreteField,
    rank: int,
    required: Extent,
    interior: Extent,
    access_pattern: AccessPattern,
    local_ids: set[ComponentId],
) -> Any:
    assert target.payload is not None and target.extent is not None  # target is a leaf
    target_payload = target.payload
    target_work = _intersect_extents(target.extent, required)
    if target_work is None:
        return target_payload

    for halo_piece in _subtract_extent(target_work, interior):
        candidates = _source_candidates(
            field=field,
            target=target,
            halo_piece=halo_piece,
            access_pattern=access_pattern,
        )
        local_candidates = [
            (source, overlap)
            for source, overlap in candidates
            if source.segment_id is not None and source.segment_id in local_ids
        ]
        if len(local_candidates) > 1:
            msg = "Multiple same-rank source segments overlap one halo region"
            raise ValueError(msg)
        if len(local_candidates) == 0:
            if candidates:
                msg = (
                    "HaloFillPolicy cannot fill a halo from rank "
                    f"{rank}; multi-rank halo exchange is not implemented"
                )
                raise NotImplementedError(msg)
            continue

        source, overlap = local_candidates[0]
        assert (
            source.extent is not None and source.payload is not None
        )  # source is a leaf
        target_payload = target_payload.at[_payload_slices(target.extent, overlap)].set(
            source.payload[_payload_slices(source.extent, overlap)]
        )

    return target_payload


def _source_candidates(
    *,
    field: DiscreteField,
    target: DiscreteField,
    halo_piece: Extent,
    access_pattern: AccessPattern,
) -> list[tuple[DiscreteField, Extent]]:
    candidates: list[tuple[DiscreteField, Extent]] = []
    for source in field.segments:
        if source.segment_id == target.segment_id:
            continue
        source_interior = _segment_interior(source, access_pattern)
        overlap = _intersect_extents(source_interior, halo_piece)
        if overlap is not None:
            candidates.append((source, overlap))
    return candidates


def _segment_interior(segment: DiscreteField, access_pattern: AccessPattern) -> Extent:
    if segment.interior_extent is not None:
        return segment.interior_extent
    assert segment.extent is not None  # segment is always a leaf at this call site
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
    """Return pieces of *extent* that are outside *removed*."""
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
    for combo in product(*axis_parts):
        slices = tuple(part for part, _inside in combo)
        if all(_inside for _part, _inside in combo):
            continue
        if all(axis_slice.start < axis_slice.stop for axis_slice in slices):
            pieces.append(Extent(slices))
    return tuple(pieces)


def _payload_slices(parent: Extent, child: Extent) -> tuple[slice, ...]:
    return tuple(
        slice(
            child_slice.start - parent_slice.start,
            child_slice.stop - parent_slice.start,
        )
        for parent_slice, child_slice in zip(parent.slices, child.slices, strict=False)
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
    "HaloFillFence",
    "HaloFillPolicy",
]
