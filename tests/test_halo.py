"""Tests for fill_halo: ghost-cell copy from interior-sized arrays."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_foundry.descriptor import AccessPattern
from cosmic_foundry.field import ContinuousField
from cosmic_foundry.mesh import discretize, fill_halo, partition_domain
from cosmic_foundry.record import Array, ComponentId


def _make_1d_mesh(n_ranks: int = 1) -> Array:
    return partition_domain(
        domain_origin=(0.0,),
        domain_size=(8.0,),
        n_cells=(8,),
        blocks_per_axis=(2,),
        n_ranks=n_ranks,
    )


def test_single_rank_fill_copies_1d_neighbor_ghosts() -> None:
    """Ghost cells at the shared face are filled from the neighbor interior."""
    mesh = _make_1d_mesh()
    access = AccessPattern((1,))

    f = ContinuousField(name="phi", fn=lambda x: x)
    field = discretize(f, mesh)

    filled = fill_halo(mesh, field, access, rank=0)

    # Patch 0 interior: [0, 4), values ~0.5, 1.5, 2.5, 3.5 (cell centers with h=1)
    # Patch 1 interior: [4, 8), values ~4.5, 5.5, 6.5, 7.5
    # Filled block 0 halo-sized array: shape (6,), interior at [1:5]
    # Right ghost (index 5) should be block 1's first interior value (4.5)
    b0 = filled[ComponentId(0)]
    b1 = filled[ComponentId(1)]

    assert b0.shape == (6,)  # halo-expanded: [-1, 5)
    assert b1.shape == (6,)  # halo-expanded: [3, 9)

    # Right ghost of block 0 = left interior of block 1 (global index 4 → 4.5)
    assert b0[5] == pytest.approx(4.5)
    # Left ghost of block 0 has no neighbor → zero-filled
    assert b0[0] == pytest.approx(0.0)
    # Left ghost of block 1 = right interior of block 0 (global index 3 → 3.5)
    assert b1[0] == pytest.approx(3.5)
    # Right ghost of block 1 has no neighbor → zero-filled
    assert b1[5] == pytest.approx(0.0)


def test_single_rank_fill_copies_2d_face_slab() -> None:
    """A full face slab is copied, not only one scalar cell."""
    mesh = partition_domain(
        domain_origin=(0.0, 0.0),
        domain_size=(2.0, 3.0),
        n_cells=(4, 3),
        blocks_per_axis=(2, 1),
        n_ranks=1,
    )
    access = AccessPattern((1, 1))

    # f(x, y) = 10*x + y so each block has distinct values
    f = ContinuousField(name="phi", fn=lambda x, y: 10.0 * x + y)
    field = discretize(f, mesh)

    filled = fill_halo(mesh, field, access, rank=0)

    # Patch 0: interior x in [0, 2), y in [0, 3) → shape (2, 3) interior, (4, 5) halo
    # Patch 1: interior x in [2, 4), y in [0, 3)
    # The right ghost slab of block 0 (halo row 3 in array) should equal
    # block 1's left interior slab (row 0 in block 1's interior array).
    b0 = filled[ComponentId(0)]
    b1_interior = field[ComponentId(1)]

    assert b0.shape == (4, 5)  # halo-expanded
    np.testing.assert_allclose(b0[3, 1:4], b1_interior[0, :])


def test_fill_halo_returns_new_array_without_mutating_original() -> None:
    mesh = _make_1d_mesh()
    access = AccessPattern((1,))

    f = ContinuousField(name="phi", fn=lambda x: x)
    field = discretize(f, mesh)
    original = field[ComponentId(0)].copy()

    filled = fill_halo(mesh, field, access, rank=0)

    assert filled is not field
    np.testing.assert_allclose(field[ComponentId(0)], original)
    assert filled[ComponentId(0)].shape == (6,)


def test_fill_halo_rejects_off_rank_neighbor_until_multi_rank_implemented() -> None:
    mesh = _make_1d_mesh(n_ranks=2)
    access = AccessPattern((1,))

    f = ContinuousField(name="phi", fn=lambda x: x)
    field = discretize(f, mesh)

    with pytest.raises(NotImplementedError, match="multi-rank"):
        fill_halo(mesh, field, access, rank=0)


def test_interior_values_preserved_after_fill() -> None:
    """Interior values in the returned array must equal the original field."""
    mesh = _make_1d_mesh()
    access = AccessPattern((1,))

    f = ContinuousField(name="phi", fn=lambda x: x * x)
    field = discretize(f, mesh)

    filled = fill_halo(mesh, field, access, rank=0)

    # Patch 0 halo extent [-1, 5): interior at array indices [1:5]
    np.testing.assert_allclose(filled[ComponentId(0)][1:5], field[ComponentId(0)])
    # Patch 1 halo extent [3, 9): interior at array indices [1:5]
    np.testing.assert_allclose(filled[ComponentId(1)][1:5], field[ComponentId(1)])
