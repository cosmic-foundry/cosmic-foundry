"""Tests for DifferentialOperator."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.theory.differential_operator import DifferentialOperator
from cosmic_foundry.theory.euclidean_space import EuclideanSpace
from cosmic_foundry.theory.function import Function
from cosmic_foundry.theory.smooth_manifold import SmoothManifold


class _Gradient(DifferentialOperator):
    @property
    def manifold(self) -> SmoothManifold:
        return EuclideanSpace(3)

    @property
    def order(self) -> int:
        return 1

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


class _Laplacian(DifferentialOperator):
    @property
    def manifold(self) -> SmoothManifold:
        return EuclideanSpace(3)

    @property
    def order(self) -> int:
        return 2

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


def test_differential_operator_is_function() -> None:
    assert issubclass(DifferentialOperator, Function)


def test_differential_operator_is_abstract() -> None:
    with pytest.raises(TypeError):
        DifferentialOperator()  # type: ignore[abstract]


def test_gradient_order() -> None:
    assert _Gradient().order == 1


def test_laplacian_order() -> None:
    assert _Laplacian().order == 2


def test_manifold_is_smooth() -> None:
    op = _Gradient()
    assert isinstance(op.manifold, SmoothManifold)
