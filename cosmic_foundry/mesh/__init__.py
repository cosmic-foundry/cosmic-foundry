"""Uniform structured mesh."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import (
    Extent,
    payload_slices,
)
from cosmic_foundry.computation.overlap import fill_by_overlap
from cosmic_foundry.theory.function import Function
from cosmic_foundry.theory.indexed_set import IndexedSet
from cosmic_foundry.theory.located_discretization import LocatedDiscretization


@dataclass(frozen=True)
class Patch(LocatedDiscretization):
    """One contiguous patch of uniformly-spaced cells — a uniform Cartesian
    LocatedDiscretization.

    Owns topology and coordinate metadata only; array payloads live in
    Array[T] for some backend array type T.  The coordinate function φ(i) = origin + i·h
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

    def intersect(self, other: IndexedSet) -> Extent | None:
        return self.index_extent.intersect(other)

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
                   patches_per_axis ∈ ℤⁿ) — a continuous domain specification
                   and discretization parameters
        codomain — Array[Patch]: a finite indexed family of Patches
                   partitioning Ω into ∏ patches_per_axis patches, each
                   covering n_cells/patches_per_axis interior cells with
                   spacing h = L/n_cells
        operator — (Ω, h, patches) ↦ {(P_i, Ω_h^int(P_i))}_i

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
        for multi_idx in itertools.product(
            *(range(blocks_per_axis[i]) for i in range(ndim))
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

        return Array(elements=tuple(patches))


partition_domain = PartitionDomain()


def covers(mesh: Array[Patch], extent: Extent) -> bool:
    """Return True iff the union of patch index_extents covers *extent*."""
    shape = extent.shape
    origin = tuple(s.start for s in extent.slices)
    covered = np.zeros(shape, dtype=bool)
    for patch in mesh.elements:
        intersection = patch.index_extent.intersect(extent)
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
    field: Array[Any],
    radii: tuple[int, ...],
) -> Array[Any]:
    """Return a new Array[T] with halo-expanded payloads.

    T is the backend array type (currently jax.Array).  Ghost cells are filled
    from patch interior neighbors.  Each element of *field* must have a shape
    equal to patch.index_extent.shape (i.e. the interior, as returned by
    discretize()).  The returned Array has halo-sized elements:
    patch.index_extent.expand(radii).shape.
    """
    import jax.numpy as jnp

    updated: dict[int, Any] = {}

    for cid in range(len(mesh.elements)):
        patch = mesh[cid]
        interior = patch.index_extent
        halo_extent = interior.expand(radii)

        expanded = jnp.zeros(halo_extent.shape, dtype=field[cid].dtype)
        expanded = expanded.at[payload_slices(halo_extent, interior)].set(field[cid])

        _validate_halo_coverage(cid, halo_extent, interior, mesh)

        for i in range(len(mesh.elements)):
            if i == cid:
                continue
            expanded = fill_by_overlap(
                mesh[i].index_extent, field[i], halo_extent, expanded
            )

        updated[cid] = expanded

    new_elements = tuple(updated.get(i, field[i]) for i in range(len(field.elements)))
    return Array(elements=new_elements)


def _validate_halo_coverage(
    target_cid: int,
    halo_extent: Extent,
    interior: Extent,
    mesh: Array[Patch],
) -> None:
    for halo_piece in _subtract_extent(halo_extent, interior):
        candidates = [
            i
            for i in range(len(mesh.elements))
            if i != target_cid
            and mesh[i].index_extent.intersect(halo_piece) is not None
        ]
        if len(candidates) > 1:
            msg = "Multiple source patches overlap one halo region"
            raise ValueError(msg)


def _subtract_extent(extent: Extent, removed: Extent) -> tuple[Extent, ...]:
    """Return pieces of *extent* that lie outside *removed*."""
    overlap = extent.intersect(removed)
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


__all__ = [
    "Patch",
    "PartitionDomain",
    "covers",
    "fill_halo",
    "partition_domain",
]
