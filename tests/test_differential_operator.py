"""Tests for DifferentialOperator."""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.foundation.function import Function


class _StubManifold(Manifold):
    @property
    def ndim(self) -> int:
        return 3

    @property
    def atlas(self) -> Any:
        raise NotImplementedError


_M = _StubManifold()


class _Gradient(DifferentialOperator):
    @property
    def manifold(self) -> Manifold:
        return _M

    @property
    def order(self) -> int:
        return 1

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


class _Laplacian(DifferentialOperator):
    @property
    def manifold(self) -> Manifold:
        return _M

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


def test_manifold_is_manifold() -> None:
    op = _Gradient()
    assert isinstance(op.manifold, Manifold)
