"""Tests for UniformGrid.fill_halo (same-rank ghost-cell copy)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from cosmic_foundry.descriptor import AccessPattern, Extent
from cosmic_foundry.mesh import DistributedField, FieldSegment, partition_domain
from cosmic_foundry.record import ComponentId, Placement


def _make_1d_grid(n_ranks: int = 1):
    return partition_domain(
        domain_origin=(0.0,),
        domain_size=(8.0,),
        n_cells=(8,),
        blocks_per_axis=(2,),
        n_ranks=n_ranks,
    )


def _segment_with_interior_values(
    segment_id: int,
    extent: Extent,
    interior: Extent,
    *,
    fill_value: float,
    offset: float,
) -> FieldSegment:
    payload = jnp.full(extent.shape, fill_value, dtype=jnp.float64)
    interior_slices = tuple(
        slice(a.start - p.start, a.stop - p.start)
        for p, a in zip(extent.slices, interior.slices, strict=False)
    )
    local_shape = tuple(a.stop - a.start for a in interior.slices)
    values = jnp.arange(local_shape[0], dtype=jnp.float64) + offset
    payload = payload.at[interior_slices].set(values)
    return FieldSegment(
        name="phi", segment_id=ComponentId(segment_id), payload=payload, extent=extent
    )


def test_single_rank_fill_copies_1d_neighbor_ghosts() -> None:
    """Same-rank block faces copy from neighboring block interiors."""
    grid = _make_1d_grid()
    access = AccessPattern((1,))
    left = _segment_with_interior_values(
        0,
        Extent((slice(-1, 5),)),
        Extent((slice(0, 4),)),
        fill_value=-10.0,
        offset=100.0,
    )
    right = _segment_with_interior_values(
        1,
        Extent((slice(3, 9),)),
        Extent((slice(4, 8),)),
        fill_value=-20.0,
        offset=200.0,
    )
    field = DistributedField(
        "phi",
        (left, right),
        Placement({ComponentId(0): 0, ComponentId(1): 0}),
    )

    filled = grid.fill_halo(field, access, rank=0)

    # Right ghost of left block filled from right block interior at global 4 → 200.0
    assert filled.segment(ComponentId(0)).payload[5] == pytest.approx(200.0)
    # Left ghost of left block has no neighbor (domain boundary) → unchanged
    assert filled.segment(ComponentId(0)).payload[0] == pytest.approx(-10.0)
    # Left ghost of right block filled from left block interior at global 3 → 103.0
    assert filled.segment(ComponentId(1)).payload[0] == pytest.approx(103.0)
    # Right ghost of right block has no neighbor (domain boundary) → unchanged
    assert filled.segment(ComponentId(1)).payload[5] == pytest.approx(-20.0)


def test_single_rank_fill_copies_2d_face_slab() -> None:
    """A full face slab is copied, not only one scalar cell."""
    grid = partition_domain(
        domain_origin=(0.0, 0.0),
        domain_size=(2.0, 3.0),
        n_cells=(4, 3),
        blocks_per_axis=(2, 1),
        n_ranks=1,
    )
    access = AccessPattern((1, 1))
    bottom_payload = jnp.full((4, 5), -1.0, dtype=jnp.float64)
    top_payload = jnp.full((4, 5), -2.0, dtype=jnp.float64)
    bottom_payload = bottom_payload.at[1:3, 1:4].set(
        jnp.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    )
    top_payload = top_payload.at[1:3, 1:4].set(
        jnp.array([[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]])
    )
    field = DistributedField(
        name="phi",
        segments=(
            FieldSegment(
                name="phi",
                segment_id=ComponentId(0),
                payload=bottom_payload,
                extent=Extent((slice(-1, 3), slice(-1, 4))),
            ),
            FieldSegment(
                name="phi",
                segment_id=ComponentId(1),
                payload=top_payload,
                extent=Extent((slice(1, 5), slice(-1, 4))),
            ),
        ),
        placement=Placement({ComponentId(0): 0, ComponentId(1): 0}),
    )

    filled = grid.fill_halo(field, access, rank=0)

    assert jnp.allclose(
        filled.segment(ComponentId(0)).payload[3, 1:4],
        jnp.array([30.0, 31.0, 32.0]),
    )


def test_fill_halo_returns_new_field_without_mutating_original() -> None:
    grid = _make_1d_grid()
    access = AccessPattern((1,))
    left = _segment_with_interior_values(
        0,
        Extent((slice(-1, 5),)),
        Extent((slice(0, 4),)),
        fill_value=-10.0,
        offset=100.0,
    )
    right = _segment_with_interior_values(
        1,
        Extent((slice(3, 9),)),
        Extent((slice(4, 8),)),
        fill_value=-20.0,
        offset=200.0,
    )
    field = DistributedField(
        "phi",
        (left, right),
        Placement({ComponentId(0): 0, ComponentId(1): 0}),
    )

    filled = grid.fill_halo(field, access, rank=0)

    assert filled is not field
    assert field.segment(ComponentId(0)).payload[5] == pytest.approx(-10.0)
    assert filled.segment(ComponentId(0)).payload[5] == pytest.approx(200.0)


def test_fill_halo_rejects_off_rank_neighbor_until_multi_rank_implemented() -> None:
    grid = _make_1d_grid(n_ranks=2)
    access = AccessPattern((1,))
    left = _segment_with_interior_values(
        0,
        Extent((slice(-1, 5),)),
        Extent((slice(0, 4),)),
        fill_value=-10.0,
        offset=100.0,
    )
    right = _segment_with_interior_values(
        1,
        Extent((slice(3, 9),)),
        Extent((slice(4, 8),)),
        fill_value=-20.0,
        offset=200.0,
    )
    field = DistributedField(
        "phi",
        (left, right),
        Placement({ComponentId(0): 0, ComponentId(1): 1}),
    )

    with pytest.raises(NotImplementedError, match="multi-rank"):
        grid.fill_halo(field, access, rank=0)
