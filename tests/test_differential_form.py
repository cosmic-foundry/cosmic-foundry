"""Tests for DifferentialForm, ScalarField, and CovectorField."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.continuous.differential_form import (
    CovectorField,
    DifferentialForm,
    ScalarField,
)
from cosmic_foundry.continuous.euclidean_space import EuclideanSpace
from cosmic_foundry.continuous.field import TensorField
from cosmic_foundry.continuous.smooth_manifold import SmoothManifold


class _Form(DifferentialForm):
    def __init__(self, degree: int) -> None:
        self._degree = degree

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def manifold(self) -> SmoothManifold:
        return EuclideanSpace(3)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


class _Scalar(ScalarField):
    @property
    def manifold(self) -> SmoothManifold:
        return EuclideanSpace(3)

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        return 0.0


class _Covector(CovectorField):
    @property
    def manifold(self) -> SmoothManifold:
        return EuclideanSpace(3)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_differential_form_is_tensor_field() -> None:
    assert issubclass(DifferentialForm, TensorField)


def test_scalar_field_is_differential_form() -> None:
    assert issubclass(ScalarField, DifferentialForm)


def test_covector_field_is_differential_form() -> None:
    assert issubclass(CovectorField, DifferentialForm)


def test_scalar_field_is_tensor_field() -> None:
    assert issubclass(ScalarField, TensorField)


def test_covector_field_is_tensor_field() -> None:
    assert issubclass(CovectorField, TensorField)


def test_differential_form_is_abstract() -> None:
    with pytest.raises(TypeError):
        DifferentialForm()  # type: ignore[abstract]


def test_scalar_field_is_abstract() -> None:
    with pytest.raises(TypeError):
        ScalarField()  # type: ignore[abstract]


def test_covector_field_is_abstract() -> None:
    with pytest.raises(TypeError):
        CovectorField()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# degree and tensor_type
# ---------------------------------------------------------------------------


def test_degree_zero_form() -> None:
    form = _Form(0)
    assert form.degree == 0
    assert form.tensor_type == (0, 0)


def test_degree_one_form() -> None:
    form = _Form(1)
    assert form.degree == 1
    assert form.tensor_type == (0, 1)


def test_degree_two_form() -> None:
    form = _Form(2)
    assert form.degree == 2
    assert form.tensor_type == (0, 2)


def test_tensor_type_tracks_degree() -> None:
    for k in range(4):
        form = _Form(k)
        assert form.tensor_type == (0, k)


def test_scalar_field_degree() -> None:
    assert _Scalar().degree == 0


def test_scalar_field_tensor_type() -> None:
    assert _Scalar().tensor_type == (0, 0)


def test_covector_field_degree() -> None:
    assert _Covector().degree == 1


def test_covector_field_tensor_type() -> None:
    assert _Covector().tensor_type == (0, 1)


# ---------------------------------------------------------------------------
# isinstance checks confirm the type relationship is real, not just nominal
# ---------------------------------------------------------------------------


def test_scalar_instance_is_differential_form() -> None:
    assert isinstance(_Scalar(), DifferentialForm)


def test_covector_instance_is_differential_form() -> None:
    assert isinstance(_Covector(), DifferentialForm)


def test_manifold_is_smooth() -> None:
    assert isinstance(_Form(2).manifold, SmoothManifold)
