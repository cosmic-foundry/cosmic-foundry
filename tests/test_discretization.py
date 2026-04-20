"""Tests for LocatedDiscretization and ModalDiscretization."""

from __future__ import annotations

import pytest

from cosmic_foundry.theory.discretization import Discretization
from cosmic_foundry.theory.euclidean_space import EuclideanSpace
from cosmic_foundry.theory.indexed_set import IndexedSet
from cosmic_foundry.theory.located_discretization import LocatedDiscretization
from cosmic_foundry.theory.modal_discretization import ModalDiscretization
from cosmic_foundry.theory.smooth_manifold import SmoothManifold

# ---------------------------------------------------------------------------
# Minimal concrete stubs
# ---------------------------------------------------------------------------


class _UniformGrid(LocatedDiscretization):
    """Minimal 1-D uniform grid on ℝ¹."""

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def ndim(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._n,)

    def intersect(self, other: IndexedSet) -> None:
        return None  # not under test

    @property
    def manifold(self) -> EuclideanSpace:
        return EuclideanSpace(1)

    def node_positions(self, axis: int) -> tuple[float, ...]:
        return tuple(float(i) for i in range(self._n))


class _FourierBasis(ModalDiscretization):
    """Minimal 1-D Fourier discretization stub."""

    def __init__(self, n_modes: int) -> None:
        self._n_modes = n_modes

    @property
    def ndim(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._n_modes,)

    def intersect(self, other: IndexedSet) -> None:
        return None  # not under test

    @property
    def basis_functions(self) -> tuple[str, ...]:
        return tuple(f"cos({k}x)" for k in range(self._n_modes))


# ---------------------------------------------------------------------------
# LocatedDiscretization hierarchy
# ---------------------------------------------------------------------------


def test_located_discretization_is_abstract() -> None:
    with pytest.raises(TypeError):
        LocatedDiscretization()  # type: ignore[abstract]


def test_located_discretization_is_discretization() -> None:
    assert issubclass(LocatedDiscretization, Discretization)
    assert issubclass(LocatedDiscretization, IndexedSet)


def test_located_discretization_manifold_is_smooth_manifold() -> None:
    grid = _UniformGrid(8)
    assert isinstance(grid.manifold, SmoothManifold)


def test_located_discretization_manifold_ndim() -> None:
    grid = _UniformGrid(8)
    assert grid.manifold.ndim == grid.ndim


def test_located_discretization_node_positions_length() -> None:
    grid = _UniformGrid(8)
    positions = grid.node_positions(0)
    assert len(positions) == grid.shape[0]


def test_located_discretization_ndim_shape_correspondence() -> None:
    grid = _UniformGrid(16)
    assert len(grid.shape) == grid.ndim


# ---------------------------------------------------------------------------
# ModalDiscretization hierarchy
# ---------------------------------------------------------------------------


def test_modal_discretization_is_abstract() -> None:
    with pytest.raises(TypeError):
        ModalDiscretization()  # type: ignore[abstract]


def test_modal_discretization_is_discretization() -> None:
    assert issubclass(ModalDiscretization, Discretization)
    assert issubclass(ModalDiscretization, IndexedSet)


def test_modal_discretization_basis_functions() -> None:
    modal = _FourierBasis(4)
    assert modal.basis_functions is not None


def test_modal_discretization_basis_cardinality() -> None:
    n = 6
    modal = _FourierBasis(n)
    assert len(modal.basis_functions) == n


def test_modal_discretization_ndim_shape_correspondence() -> None:
    modal = _FourierBasis(8)
    assert len(modal.shape) == modal.ndim
