"""Tests for BoundaryCondition, LocalBoundaryCondition, NonLocalBoundaryCondition."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.continuous.differential_form import ScalarField
from cosmic_foundry.continuous.euclidean_space import EuclideanSpace
from cosmic_foundry.continuous.field import Field
from cosmic_foundry.continuous.local_boundary_condition import LocalBoundaryCondition
from cosmic_foundry.continuous.manifold_with_boundary import ManifoldWithBoundary
from cosmic_foundry.continuous.non_local_boundary_condition import (
    NonLocalBoundaryCondition,
)
from cosmic_foundry.continuous.smooth_manifold import SmoothManifold
from cosmic_foundry.foundation.function import Function

# ---------------------------------------------------------------------------
# Minimal concrete stubs
# ---------------------------------------------------------------------------


class _Face(ManifoldWithBoundary):
    """Codimension-1 face stub."""

    def __init__(self, ndim: int) -> None:
        self._ndim = ndim

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def boundary(self) -> tuple[ManifoldWithBoundary, ...]:
        return ()


class _Box(ManifoldWithBoundary):
    """3-D box with 6 faces."""

    @property
    def ndim(self) -> int:
        return 3

    @property
    def boundary(self) -> tuple[ManifoldWithBoundary, ...]:
        return tuple(_Face(2) for _ in range(6))


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


class _FaceIdentification(NonLocalBoundaryCondition[Any, None]):
    """Periodic BC: identifies two boundary faces. Carries its own geometry."""

    def __init__(self, face_a: ManifoldWithBoundary, face_b: ManifoldWithBoundary):
        self.faces = (face_a, face_b)

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


def test_face_identification_faces() -> None:
    box = _Box()
    faces = box.boundary
    bc = _FaceIdentification(faces[0], faces[1])
    assert len(bc.faces) == 2
    assert all(isinstance(f, ManifoldWithBoundary) for f in bc.faces)


def test_face_identification_codimension() -> None:
    box = _Box()
    faces = box.boundary
    bc = _FaceIdentification(faces[0], faces[1])
    for face in bc.faces:
        assert face.ndim == box.ndim - 1
