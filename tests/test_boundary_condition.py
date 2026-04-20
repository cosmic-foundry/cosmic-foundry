"""Tests for BoundaryCondition, LocalBoundaryCondition, NonLocalBoundaryCondition."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.theory.boundary_condition import BoundaryCondition
from cosmic_foundry.theory.field import Field
from cosmic_foundry.theory.function import Function
from cosmic_foundry.theory.local_boundary_condition import LocalBoundaryCondition
from cosmic_foundry.theory.manifold_with_boundary import ManifoldWithBoundary
from cosmic_foundry.theory.non_local_boundary_condition import (
    NonLocalBoundaryCondition,
)

# ---------------------------------------------------------------------------
# Minimal concrete stubs for testing ABC structure
# ---------------------------------------------------------------------------


class _ConstantField(Field):
    name: str = "constant"

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return None


class _FaceIdentification(NonLocalBoundaryCondition):
    """Periodic BC: identifies two boundary faces. Carries its own geometry."""

    def __init__(self, face_a: ManifoldWithBoundary, face_b: ManifoldWithBoundary):
        self.faces = (face_a, face_b)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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


def test_face_identification_faces(euclidean_domain_3d: Any) -> None:
    faces = euclidean_domain_3d.boundary
    bc = _FaceIdentification(faces[0], faces[1])
    assert len(bc.faces) == 2
    assert all(isinstance(f, ManifoldWithBoundary) for f in bc.faces)


def test_face_identification_codimension(euclidean_domain_3d: Any) -> None:
    faces = euclidean_domain_3d.boundary
    bc = _FaceIdentification(faces[0], faces[1])
    for face in bc.faces:
        assert face.ndim == euclidean_domain_3d.ndim - 1
