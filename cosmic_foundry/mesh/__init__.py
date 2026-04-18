"""Uniform structured mesh."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from cosmic_foundry.descriptor import AccessPattern, Extent
from cosmic_foundry.field import ContinuousField, PatchFunction
from cosmic_foundry.function import Function
from cosmic_foundry.located_discretization import LocatedDiscretization
from cosmic_foundry.record import Array, ComponentId, Placement


@dataclass(frozen=True)
class Patch(LocatedDiscretization):
    """One contiguous patch of uniformly-spaced cells — a uniform Cartesian
    LocatedDiscretization.

    Owns topology and coordinate metadata only; array payloads live in
    Array[PatchFunction].  The coordinate function φ(i) = origin + i·h
    maps each cell index to its center position in physical space.
    """

    index_extent: Extent
    origin: tuple[float, ...]  # physical coord of first cell center
    cell_spacing: tuple[float, ...]  # h_i along each axis

    @property
    def ndim(self) -> int:
        return self.index_extent.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.index_extent.shape

    def node_positions(self, axis: int) -> Any:
        """1-D coordinate array of cell-center positions along *axis*."""
        import jax.numpy as jnp

        return (
            self.origin[axis] + jnp.arange(self.shape[axis]) * self.cell_spacing[axis]
        )


@dataclass(frozen=True)
class PartitionDomain(Function):
    """Partition a continuous domain into a discrete patch grid.

    Function:
        domain   — (Ω = ∏ᵢ [oᵢ, oᵢ+Lᵢ], n_cells ∈ ℤⁿ,
                   patches_per_axis ∈ ℤⁿ, n_ranks ∈ ℤ) — a continuous
                   domain specification and discretization parameters
        codomain — Array[Patch]: a finite indexed family of Patches
                   partitioning Ω into ∏ patches_per_axis patches, each
                   covering n_cells/patches_per_axis interior cells with
                   spacing h = L/n_cells, assigned to ranks round-robin
        operator — (Ω, h, patches) ↦ {(P_i, Ω_h^int(P_i), rank_i)}_i

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
    ) -> Array[Patch]:
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

        patches: list[Patch] = []
        owners: dict[ComponentId, int] = {}

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
            patches.append(
                Patch(
                    index_extent=index_extent,
                    origin=origin,
                    cell_spacing=h,
                )
            )
            owners[ComponentId(flat_id)] = flat_id % n_ranks

        return Array(elements=tuple(patches), placement=Placement(owners))


partition_domain = PartitionDomain()


def discretize(f: ContinuousField, mesh: Array[Patch]) -> Array[PatchFunction]:
    """Sample *f* at each patch's node positions, returning Array[PatchFunction].

    The returned Array has the same Placement as the mesh: element i is the
    PatchFunction on patch i.  Each payload has shape equal to the patch's
    index_extent.shape and no ghost cells.
    """
    import jax.numpy as jnp

    elements: list[PatchFunction] = []
    for patch in mesh.elements:
        axes = [patch.node_positions(axis) for axis in range(patch.ndim)]
        coords = jnp.meshgrid(*axes, indexing="ij")
        payload = f.sample(*coords).payload
        elements.append(PatchFunction(name=f.name, payload=payload))
    return Array(elements=tuple(elements), placement=mesh.placement)


def covers(mesh: Array[Patch], extent: Extent) -> bool:
    """Return True iff the union of patch index_extents covers *extent*."""
    shape = extent.shape
    origin = tuple(s.start for s in extent.slices)
    covered = np.zeros(shape, dtype=bool)
    for patch in mesh.elements:
        intersection = _intersect_extents(patch.index_extent, extent)
        if intersection is None:
            continue
        local_idx = tuple(
            slice(s.start - o, s.stop - o)
            for s, o in zip(intersection.slices, origin, strict=False)
        )
        covered[local_idx] = True
    return bool(covered.all())


def fill_halo(
    mesh: Array[Patch],
    field: Array[PatchFunction],
    access_pattern: AccessPattern,
    rank: int,
) -> Array[PatchFunction]:
    """Return a new Array[PatchFunction] with halo-expanded payloads.

    Ghost cells are filled from same-rank neighbor interiors.
    Each element of *field* must have a payload whose shape equals
    patch.index_extent.shape (i.e. the interior, as returned by discretize()).
    The returned Array has halo-sized payloads:
    patch.index_extent.expand(access_pattern).shape.  Ghost-cell values are
    copied from the interior of neighboring patches owned by the same rank.
    Multi-rank halo exchange is not implemented; a NotImplementedError is
    raised if a ghost cell's source lives on a different rank.
    """
    import jax.numpy as jnp

    local_ids = mesh.placement.segments_for_rank(rank)
    updated: dict[ComponentId, PatchFunction] = {}
    for cid in local_ids:
        patch = mesh[cid]
        interior = patch.index_extent
        halo_extent = interior.expand(access_pattern)

        expanded = jnp.zeros(halo_extent.shape, dtype=field[cid].payload.dtype)
        expanded = expanded.at[_payload_slices(halo_extent, interior)].set(
            field[cid].payload
        )

        updated_payload = _fill_patch_halo(
            target_cid=cid,
            target_interior=interior,
            target_halo_extent=halo_extent,
            expanded_payload=expanded,
            mesh=mesh,
            field=field,
            rank=rank,
            local_ids=local_ids,
        )
        updated[cid] = PatchFunction(name=field[cid].name, payload=updated_payload)
    new_elements = tuple(
        updated.get(ComponentId(i), field[ComponentId(i)])
        for i in range(len(field.elements))
    )
    return Array(elements=new_elements, placement=field.placement)


def _fill_patch_halo(
    *,
    target_cid: ComponentId,
    target_interior: Extent,
    target_halo_extent: Extent,
    expanded_payload: Any,
    mesh: Array[Patch],
    field: Array[PatchFunction],
    rank: int,
    local_ids: frozenset[ComponentId],
) -> Any:
    payload = expanded_payload
    for halo_piece in _subtract_extent(target_halo_extent, target_interior):
        candidates: list[tuple[ComponentId, Extent]] = []
        for i in range(len(mesh.elements)):
            src_cid = ComponentId(i)
            if src_cid == target_cid:
                continue
            overlap = _intersect_extents(mesh[src_cid].index_extent, halo_piece)
            if overlap is not None:
                candidates.append((src_cid, overlap))

        local_candidates = [(c, ov) for c, ov in candidates if c in local_ids]

        if len(local_candidates) > 1:
            msg = "Multiple same-rank source patches overlap one halo region"
            raise ValueError(msg)
        if len(local_candidates) == 0:
            if candidates:
                msg = (
                    f"fill_halo cannot fill from rank {rank}; "
                    "multi-rank halo exchange is not implemented"
                )
                raise NotImplementedError(msg)
            continue

        src_cid, overlap = local_candidates[0]
        src_interior = mesh[src_cid].index_extent
        payload = payload.at[_payload_slices(target_halo_extent, overlap)].set(
            field[src_cid].payload[_payload_slices(src_interior, overlap)]
        )

    return payload


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
    "Patch",
    "PartitionDomain",
    "covers",
    "discretize",
    "fill_halo",
    "partition_domain",
]
