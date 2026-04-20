"""Tests for BoundaryCondition, LocalBoundaryCondition, NonLocalBoundaryCondition."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.continuous.differential_form import ScalarField
from cosmic_foundry.continuous.euclidean_space import EuclideanSpace
from cosmic_foundry.continuous.field import Field
from cosmic_foundry.continuous.local_boundary_condition import LocalBoundaryCondition
from cosmic_foundry.continuous.non_local_boundary_condition import (
    NonLocalBoundaryCondition,
)
from cosmic_foundry.continuous.smooth_manifold import SmoothManifold
from cosmic_foundry.foundation.function import Function

# ---------------------------------------------------------------------------
# Minimal concrete stubs
# ---------------------------------------------------------------------------


class _ConstantField(ScalarField):
    @property
    def manifold(self) -> SmoothManifold:
        return EuclideanSpace(3)

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


class _DirichletBC(LocalBoundaryCondition[Any, None]):
    @property
    def alpha(self) -> float:
        return 1.0

    @property
    def beta(self) -> float:
        return 0.0

    @property
    def constraint(self) -> Field:
        return _ConstantField()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


class _NeumannBC(LocalBoundaryCondition[Any, None]):
    @property
    def alpha(self) -> float:
        return 0.0

    @property
    def beta(self) -> float:
        return 1.0

    @property
    def constraint(self) -> Field:
        return _ConstantField()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


class _PeriodicBC(NonLocalBoundaryCondition[Any, None]):
    """Periodic BC: identifies face_a with face_b separated by period L.

    Representative of the minimal geometry a real periodic BC carries:
    which two faces are identified, and the translation distance between them.
    Concrete computation/ subclasses will use these to copy ghost cells.
    """

    def __init__(self, face_a: str, face_b: str, period: float) -> None:
        self.face_a = face_a
        self.face_b = face_b
        self.period = period

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Hierarchy tests
# ---------------------------------------------------------------------------


def test_boundary_condition_is_function() -> None:
    assert issubclass(BoundaryCondition, Function)


def test_local_is_boundary_condition() -> None:
    assert issubclass(LocalBoundaryCondition, BoundaryCondition)


def test_non_local_is_boundary_condition() -> None:
    assert issubclass(NonLocalBoundaryCondition, BoundaryCondition)


def test_boundary_condition_is_abstract() -> None:
    with pytest.raises(TypeError):
        BoundaryCondition()  # type: ignore[abstract]


def test_local_boundary_condition_is_abstract() -> None:
    with pytest.raises(TypeError):
        LocalBoundaryCondition()  # type: ignore[abstract]


def test_non_local_boundary_condition_is_abstract() -> None:
    with pytest.raises(TypeError):
        NonLocalBoundaryCondition()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# LocalBoundaryCondition property tests
# ---------------------------------------------------------------------------


def test_dirichlet_coefficients() -> None:
    bc = _DirichletBC()
    assert bc.alpha == 1.0
    assert bc.beta == 0.0


def test_neumann_coefficients() -> None:
    bc = _NeumannBC()
    assert bc.alpha == 0.0
    assert bc.beta == 1.0


def test_local_constraint_is_field() -> None:
    bc = _DirichletBC()
    assert isinstance(bc.constraint, Field)


# ---------------------------------------------------------------------------
# NonLocalBoundaryCondition — concrete subclass carries its own geometry
# ---------------------------------------------------------------------------


def test_periodic_bc_is_non_local() -> None:
    bc = _PeriodicBC("xmin", "xmax", period=1.0)
    assert isinstance(bc, NonLocalBoundaryCondition)
    assert isinstance(bc, BoundaryCondition)


def test_periodic_bc_is_not_local() -> None:
    bc = _PeriodicBC("xmin", "xmax", period=1.0)
    assert not isinstance(bc, LocalBoundaryCondition)


def test_periodic_bc_identifies_faces() -> None:
    bc = _PeriodicBC("xmin", "xmax", period=1.0)
    assert bc.face_a == "xmin"
    assert bc.face_b == "xmax"


def test_periodic_bc_carries_period() -> None:
    bc = _PeriodicBC("xmin", "xmax", period=2.5)
    assert bc.period == 2.5
