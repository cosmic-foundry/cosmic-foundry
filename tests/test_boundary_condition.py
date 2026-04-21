"""Tests for the Constraint ABC and BoundaryCondition hierarchy."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.continuous.differential_form import DifferentialForm
from cosmic_foundry.continuous.field import Field
from cosmic_foundry.continuous.local_boundary_condition import LocalBoundaryCondition
from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.continuous.non_local_boundary_condition import (
    NonLocalBoundaryCondition,
)

# ---------------------------------------------------------------------------
# Minimal concrete stubs
# ---------------------------------------------------------------------------


class _StubManifold(Manifold):
    def __init__(self, ndim: int) -> None:
        self._ndim = ndim

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def atlas(self) -> Any:
        raise NotImplementedError


_BOUNDARY = _StubManifold(2)
_DOMAIN = _StubManifold(3)


class _ConstantField(DifferentialForm):
    @property
    def degree(self) -> int:
        return 0

    @property
    def manifold(self) -> Manifold:
        return _DOMAIN

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


class _DirichletBC(LocalBoundaryCondition):
    @property
    def alpha(self) -> float:
        return 1.0

    @property
    def beta(self) -> float:
        return 0.0

    @property
    def constraint(self) -> Field:
        return _ConstantField()


class _NeumannBC(LocalBoundaryCondition):
    @property
    def alpha(self) -> float:
        return 0.0

    @property
    def beta(self) -> float:
        return 1.0

    @property
    def constraint(self) -> Field:
        return _ConstantField()


class _PeriodicBC(NonLocalBoundaryCondition):
    """Periodic BC: identifies face_a with face_b separated by period L."""

    def __init__(self, face_a: str, face_b: str, period: float) -> None:
        self.face_a = face_a
        self.face_b = face_b
        self.period = period

    @property
    def support(self) -> Manifold:
        return _BOUNDARY


# ---------------------------------------------------------------------------
# Constraint.support
# ---------------------------------------------------------------------------


def test_dirichlet_support_is_manifold() -> None:
    bc = _DirichletBC()
    assert isinstance(bc.support, Manifold)


def test_periodic_support_is_manifold() -> None:
    bc = _PeriodicBC("xmin", "xmax", period=1.0)
    assert isinstance(bc.support, Manifold)


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
