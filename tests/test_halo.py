"""Tests for the single-rank HaloFillPolicy."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from cosmic_foundry.fields import DiscreteField, Placement
from cosmic_foundry.halo import HaloFillFence, HaloFillPolicy
from cosmic_foundry.kernels import ComponentId, Extent, Region, Stencil


def _segment_with_interior_values(
    segment_id: int,
    extent: Extent,
    interior: Extent,
    *,
    fill_value: float,
    offset: float,
) -> DiscreteField:
    payload = jnp.full(extent.shape, fill_value, dtype=jnp.float64)
    interior_slices = tuple(
        slice(axis.start - parent.start, axis.stop - parent.start)
        for parent, axis in zip(extent.slices, interior.slices, strict=False)
    )
    local_shape = tuple(axis.stop - axis.start for axis in interior.slices)
    values = jnp.arange(local_shape[0], dtype=jnp.float64) + offset
    payload = payload.at[interior_slices].set(values)
    return DiscreteField(
        name="phi", segment_id=ComponentId(segment_id), payload=payload, extent=extent
    )


def test_single_rank_fill_copies_1d_neighbor_ghosts() -> None:
    """Same-rank block faces copy from neighboring block interiors."""
    access = Stencil((1,))
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
    field = DiscreteField(
        "phi",
        (left, right),
        Placement({ComponentId(0): 0, ComponentId(1): 0}),
    )

    policy = HaloFillPolicy()
    filled_left = policy.execute(
        HaloFillFence(field, Region(Extent((slice(0, 4),))), access),
        rank=0,
    )
    filled_right = policy.execute(
        HaloFillFence(field, Region(Extent((slice(4, 8),))), access),
        rank=0,
    )

    assert filled_left.segment(ComponentId(0)).payload[5] == pytest.approx(200.0)
    assert filled_left.segment(ComponentId(0)).payload[0] == pytest.approx(-10.0)
    assert filled_right.segment(ComponentId(1)).payload[0] == pytest.approx(103.0)
    assert filled_right.segment(ComponentId(1)).payload[5] == pytest.approx(-20.0)


def test_single_rank_fill_copies_2d_face_slab() -> None:
    """A full face slab is copied, not only one scalar cell."""
    access = Stencil((1, 1))
    bottom_payload = jnp.full((4, 5), -1.0, dtype=jnp.float64)
    top_payload = jnp.full((4, 5), -2.0, dtype=jnp.float64)
    bottom_payload = bottom_payload.at[1:3, 1:4].set(
        jnp.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    )
    top_payload = top_payload.at[1:3, 1:4].set(
        jnp.array([[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]])
    )
    field = DiscreteField(
        name="phi",
        segments=(
            DiscreteField(
                name="phi",
                segment_id=ComponentId(0),
                payload=bottom_payload,
                extent=Extent((slice(-1, 3), slice(-1, 4))),
            ),
            DiscreteField(
                name="phi",
                segment_id=ComponentId(1),
                payload=top_payload,
                extent=Extent((slice(1, 5), slice(-1, 4))),
            ),
        ),
        placement=Placement({ComponentId(0): 0, ComponentId(1): 0}),
    )

    filled = HaloFillPolicy().execute(
        HaloFillFence(field, Region(Extent((slice(0, 2), slice(0, 3)))), access),
        rank=0,
    )

    assert jnp.allclose(
        filled.segment(ComponentId(0)).payload[3, 1:4],
        jnp.array([30.0, 31.0, 32.0]),
    )


def test_execute_returns_new_field_without_mutating_original() -> None:
    access = Stencil((1,))
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
    field = DiscreteField(
        "phi",
        (left, right),
        Placement({ComponentId(0): 0, ComponentId(1): 0}),
    )

    filled = HaloFillPolicy().execute(
        HaloFillFence(field, Region(Extent((slice(0, 4),))), access),
        rank=0,
    )

    assert filled is not field
    assert field.segment(ComponentId(0)).payload[5] == pytest.approx(-10.0)
    assert filled.segment(ComponentId(0)).payload[5] == pytest.approx(200.0)


def test_execute_rejects_required_extent_not_covered() -> None:
    access = Stencil((1,))
    segment = DiscreteField(
        name="phi",
        segment_id=ComponentId(0),
        payload=jnp.zeros((4,), dtype=jnp.float64),
        extent=Extent((slice(0, 4),)),
    )
    field = DiscreteField(
        name="phi", segments=(segment,), placement=Placement({ComponentId(0): 0})
    )

    with pytest.raises(ValueError, match="Region plus halo"):
        HaloFillPolicy().execute(
            HaloFillFence(field, Region(Extent((slice(0, 4),))), access),
            rank=0,
        )


def test_execute_rejects_off_rank_neighbor_until_multi_rank_policy_exists() -> None:
    access = Stencil((1,))
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
    field = DiscreteField(
        "phi",
        (left, right),
        Placement({ComponentId(0): 0, ComponentId(1): 1}),
    )

    with pytest.raises(NotImplementedError, match="multi-rank"):
        HaloFillPolicy().execute(
            HaloFillFence(field, Region(Extent((slice(0, 4),))), access),
            rank=0,
        )
