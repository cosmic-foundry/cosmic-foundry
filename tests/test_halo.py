"""Tests for fill_halo: ghost-cell copy from interior-sized arrays."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.field import discretize
from cosmic_foundry.geometry.domain import Domain
from cosmic_foundry.geometry.euclidean_space import EuclideanSpace
from cosmic_foundry.mesh import fill_halo, partition_domain


def _make_1d_mesh() -> Array:
    return partition_domain.execute(
        domain=Domain(manifold=EuclideanSpace(1), origin=(0.0,), size=(8.0,)),
        n_cells=(8,),
        blocks_per_axis=(2,),
    )


def test_single_rank_fill_copies_1d_neighbor_ghosts() -> None:
    """Ghost cells at the shared face are filled from the neighbor interior."""
    mesh = _make_1d_mesh()
    access = (1,)

    field = discretize(lambda x: x, mesh)

    filled = fill_halo(mesh, field, access)

    # Patch 0 interior: [0, 4), values ~0.5, 1.5, 2.5, 3.5 (cell centers with h=1)
    # Patch 1 interior: [4, 8), values ~4.5, 5.5, 6.5, 7.5
    # Filled block 0 halo-sized array: shape (6,), interior at [1:5]
    # Right ghost (index 5) should be block 1's first interior value (4.5)
    b0 = filled[0]
    b1 = filled[1]

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
    mesh = partition_domain.execute(
        domain=Domain(manifold=EuclideanSpace(2), origin=(0.0, 0.0), size=(2.0, 3.0)),
        n_cells=(4, 3),
        blocks_per_axis=(2, 1),
    )
    access = (1, 1)

    # f(x, y) = 10*x + y so each block has distinct values
    field = discretize(lambda x, y: 10.0 * x + y, mesh)

    filled = fill_halo(mesh, field, access)

    # Patch 0: interior x in [0, 2), y in [0, 3) → shape (2, 3) interior, (4, 5) halo
    # Patch 1: interior x in [2, 4), y in [0, 3)
    # The right ghost slab of block 0 (halo row 3 in array) should equal
    # block 1's left interior slab (row 0 in block 1's interior array).
    b0 = filled[0]
    b1_interior = field[1]

    assert b0.shape == (4, 5)  # halo-expanded
    np.testing.assert_allclose(b0[3, 1:4], b1_interior[0, :])


def test_fill_halo_returns_new_array_without_mutating_original() -> None:
    mesh = _make_1d_mesh()
    access = (1,)

    field = discretize(lambda x: x, mesh)
    original = field[0].copy()

    filled = fill_halo(mesh, field, access)

    assert filled is not field
    np.testing.assert_allclose(field[0], original)
    assert filled[0].shape == (6,)


def test_interior_values_preserved_after_fill() -> None:
    """Interior values in the returned array must equal the original field."""
    mesh = _make_1d_mesh()
    access = (1,)

    field = discretize(lambda x: x * x, mesh)

    filled = fill_halo(mesh, field, access)

    # Patch 0 halo extent [-1, 5): interior at array indices [1:5]
    np.testing.assert_allclose(filled[0][1:5], field[0])
    # Patch 1 halo extent [3, 9): interior at array indices [1:5]
    np.testing.assert_allclose(filled[1][1:5], field[1])
