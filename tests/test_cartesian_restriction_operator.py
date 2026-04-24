"""Tests for CartesianMesh.boundary() and CartesianRestrictionOperator."""

from __future__ import annotations

from typing import Any

import pytest
import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianRestrictionOperator,
)
from cosmic_foundry.theory.continuous.symbolic_function import SymbolicFunction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_x, _y = sympy.symbols("x y")
_ZERO = sympy.Integer(0)
_H = sympy.Symbol("h", positive=True)


class _Func1D(SymbolicFunction[Any, sympy.Expr]):
    # Any as M because CartesianRestrictionOperator calls f.expr and f.symbols
    # directly, bypassing the Point interface — these helpers have no manifold.
    # Once RestrictionOperator.__call__ is reparameterized in C3, these will
    # carry a real manifold type and route through field(Point(...)) normally.
    def __init__(self, expr: sympy.Expr, sym: sympy.Symbol) -> None:
        self._expr = expr
        self._sym = sym

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return (self._sym,)


class _Func2D(SymbolicFunction[Any, sympy.Expr]):
    # Any as M — same reasoning as _Func1D above.
    def __init__(self, expr: sympy.Expr) -> None:
        self._expr = expr

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return (_x, _y)


# ---------------------------------------------------------------------------
# CartesianMesh.boundary() — top-dimensional cells
# ---------------------------------------------------------------------------

# Parameters: (shape, cell_idx, expected_faces)
# Each face is (axis, face_idx, sign).
_BOUNDARY_CASES: list[
    tuple[tuple[int, ...], tuple[int, ...], list[tuple[int, tuple[int, ...], int]]]
] = [
    (
        (3,),
        (1,),
        [(0, (1,), -1), (0, (2,), +1)],
    ),
    (
        (3,),
        (0,),
        [(0, (0,), -1), (0, (1,), +1)],
    ),
    (
        (2, 2),
        (0, 1),
        [
            (0, (0, 1), -1),
            (0, (1, 1), +1),
            (1, (0, 1), -1),
            (1, (0, 2), +1),
        ],
    ),
]


@pytest.mark.parametrize("shape,idx,expected", _BOUNDARY_CASES)
def test_boundary_map_top_cells(
    shape: tuple[int, ...],
    idx: tuple[int, ...],
    expected: list[tuple[int, tuple[int, ...], int]],
) -> None:
    mesh = CartesianMesh(
        origin=tuple(_ZERO for _ in shape),
        spacing=tuple(_H for _ in shape),
        shape=shape,
    )
    ndim = len(shape)
    bmap = mesh.boundary(ndim)
    assert bmap(idx) == expected


# ---------------------------------------------------------------------------
# CartesianRestrictionOperator — cell averages
# ---------------------------------------------------------------------------


def test_cell_average_linear_1d() -> None:
    """Cell average of f=x over [a, a+h] is a + h/2 (cell midpoint).

    Verifies the limit construction lo = origin + idx * spacing and that
    the operator correctly integrates a linear function over each cell.
    """
    a = sympy.Symbol("a")
    mesh = CartesianMesh(origin=(a,), spacing=(_H,), shape=(3,))
    rh = CartesianRestrictionOperator(mesh)
    rf = rh(_Func1D(_x, _x))
    for i in range(3):
        expected = a + (_H * i + _H / 2)
        assert sympy.simplify(rf((i,)) - expected) == 0  # type: ignore[arg-type]


def test_cell_average_quadratic_1d() -> None:
    """Cell average of f=x² over [i*h, (i+1)*h] from origin=0.

    ∫_{i*h}^{(i+1)*h} x² dx / h = h²(i² + i + 1/3)
    """
    mesh = CartesianMesh(origin=(_ZERO,), spacing=(_H,), shape=(2,))
    rh = CartesianRestrictionOperator(mesh)
    rf = rh(_Func1D(_x**2, _x))
    for i in range(2):
        expected = _H**2 * (i**2 + i + sympy.Rational(1, 3))
        assert sympy.simplify(rf((i,)) - expected) == 0  # type: ignore[arg-type]


def test_cell_average_separable_2d() -> None:
    """Cell average of f=x*y factors as (avg x)(avg y).

    Verifies that the 2D integration loop iterates axes independently and
    that non-uniform spacing (hx ≠ hy) and non-zero origins are handled.
    """
    a, b = sympy.symbols("a b")
    hx, hy = sympy.symbols("hx hy", positive=True)
    mesh = CartesianMesh(origin=(a, b), spacing=(hx, hy), shape=(2, 2))
    rh = CartesianRestrictionOperator(mesh)
    rf = rh(_Func2D(_x * _y))
    for i in range(2):
        for j in range(2):
            cx = a + hx * i + hx / 2
            cy = b + hy * j + hy / 2
            assert sympy.simplify(rf((i, j)) - cx * cy) == 0  # type: ignore[arg-type]
