"""Tests for the Discretization ABC hierarchy and Patch concrete class."""

from __future__ import annotations

import pytest

from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.geometry.euclidean_space import EuclideanSpace
from cosmic_foundry.mesh import Patch
from cosmic_foundry.theory.discretization import Discretization
from cosmic_foundry.theory.indexed_set import IndexedSet
from cosmic_foundry.theory.located_discretization import LocatedDiscretization
from cosmic_foundry.theory.modal_discretization import ModalDiscretization
from cosmic_foundry.theory.set import Set
from cosmic_foundry.theory.smooth_manifold import SmoothManifold

# ---------------------------------------------------------------------------
# Instantiation guards
# ---------------------------------------------------------------------------


def test_located_discretization_is_abstract() -> None:
    with pytest.raises(TypeError):
        LocatedDiscretization()  # type: ignore[abstract]


def test_modal_discretization_is_abstract() -> None:
    with pytest.raises(TypeError):
        ModalDiscretization()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Patch isinstance chain
# ---------------------------------------------------------------------------


def test_patch_isinstance_chain() -> None:
    patch = Patch(
        manifold=EuclideanSpace(1),
        index_extent=Extent((slice(0, 4),)),
        origin=(0.125,),
        cell_spacing=(0.25,),
    )
    assert isinstance(patch, Patch)
    assert isinstance(patch, LocatedDiscretization)
    assert isinstance(patch, Discretization)
    assert isinstance(patch, IndexedSet)
    assert isinstance(patch, Set)


def test_patch_not_smooth_manifold() -> None:
    assert not issubclass(Patch, SmoothManifold)


def test_patch_not_modal_discretization() -> None:
    assert not issubclass(Patch, ModalDiscretization)


# ---------------------------------------------------------------------------
# Patch satisfies LocatedDiscretization interface
# ---------------------------------------------------------------------------


def test_patch_node_positions() -> None:
    import numpy as np

    patch = Patch(
        manifold=EuclideanSpace(1),
        index_extent=Extent((slice(0, 4),)),
        origin=(0.125,),
        cell_spacing=(0.25,),
    )
    positions = np.asarray(patch.node_positions(0))
    np.testing.assert_allclose(positions, [0.125, 0.375, 0.625, 0.875])


def test_patch_ndim_and_shape() -> None:
    patch = Patch(
        manifold=EuclideanSpace(3),
        index_extent=Extent((slice(0, 8), slice(0, 8), slice(0, 8))),
        origin=(0.0625, 0.0625, 0.0625),
        cell_spacing=(0.125, 0.125, 0.125),
    )
    assert patch.ndim == 3
    assert patch.shape == (8, 8, 8)


# ---------------------------------------------------------------------------
# Discretization branch disjoint from manifold branch
# ---------------------------------------------------------------------------


def test_discretization_disjoint_from_manifold() -> None:
    assert not issubclass(Discretization, SmoothManifold)
    assert not issubclass(SmoothManifold, Discretization)
