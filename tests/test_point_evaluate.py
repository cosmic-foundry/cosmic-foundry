"""Tests for Point[M] typed field evaluation via Field.__call__."""

from __future__ import annotations

import pytest
import sympy

from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.continuous.field import Field
from cosmic_foundry.theory.continuous.point import Point


def _euclidean(ndim: int) -> EuclideanManifold:
    return EuclideanManifold(ndim)


class _ScalarField(Field[EuclideanManifold, sympy.Expr]):
    """Minimal concrete Field for testing: f(x, y) = x² + y²."""

    def __init__(self, manifold: EuclideanManifold) -> None:
        self._manifold = manifold
        x, y = manifold.symbols
        self._expr = x**2 + y**2

    @property
    def manifold(self) -> EuclideanManifold:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._manifold.symbols


def test_call_returns_correct_value() -> None:
    m = _euclidean(2)
    field = _ScalarField(m)
    chart = m.atlas[0]
    p = Point(manifold=m, chart=chart, coords=(3, 4))
    assert field(p) == 25


def test_call_rejects_mismatched_chart() -> None:
    m2 = _euclidean(2)
    m3 = _euclidean(3)
    field = _ScalarField(m2)
    wrong_chart = m3.atlas[0]
    p = Point(manifold=m3, chart=wrong_chart, coords=(1, 2, 3))
    with pytest.raises(ValueError, match="Chart mismatch"):
        field(p)
