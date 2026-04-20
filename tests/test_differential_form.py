"""Tests for DifferentialForm."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.theory.differential_form import DifferentialForm
from cosmic_foundry.theory.euclidean_space import EuclideanSpace
from cosmic_foundry.theory.field import TensorField
from cosmic_foundry.theory.smooth_manifold import SmoothManifold


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


def test_differential_form_is_tensor_field() -> None:
    assert issubclass(DifferentialForm, TensorField)


def test_differential_form_is_abstract() -> None:
    with pytest.raises(TypeError):
        DifferentialForm()  # type: ignore[abstract]


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


def test_manifold_is_smooth() -> None:
    form = _Form(2)
    assert isinstance(form.manifold, SmoothManifold)
