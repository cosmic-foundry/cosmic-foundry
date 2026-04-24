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
_ONE = sympy.Integer(1)
_H = sympy.Symbol("h", positive=True)


class _Func1D(SymbolicFunction[Any, sympy.Expr]):
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
def assert_boundary_map_top_cells(
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


@pytest.mark.parametrize("shape,idx,expected", _BOUNDARY_CASES)
def test_boundary_map_top_cells(
    shape: tuple[int, ...],
    idx: tuple[int, ...],
    expected: list[tuple[int, tuple[int, ...], int]],
) -> None:
    assert_boundary_map_top_cells(shape, idx, expected)


def assert_boundary_lower_k_not_implemented() -> None:
    mesh = CartesianMesh(
        origin=(_ZERO, _ZERO),
        spacing=(_H, _H),
        shape=(2, 2),
    )
    with pytest.raises(NotImplementedError):
        mesh.boundary(1)


def test_boundary_lower_k_not_implemented() -> None:
    assert_boundary_lower_k_not_implemented()


def assert_boundary_out_of_range_raises() -> None:
    mesh = CartesianMesh(
        origin=(_ZERO,),
        spacing=(_H,),
        shape=(3,),
    )
    with pytest.raises(IndexError):
        mesh.boundary(0)
    with pytest.raises(IndexError):
        mesh.boundary(5)


def test_boundary_out_of_range_raises() -> None:
    assert_boundary_out_of_range_raises()


# ---------------------------------------------------------------------------
# CartesianRestrictionOperator — cell averages
# ---------------------------------------------------------------------------


def assert_cell_average_constant_1d() -> None:
    """Cell average of a constant equals the constant."""
    c = sympy.Rational(7, 3)
    mesh = CartesianMesh(origin=(_ZERO,), spacing=(_H,), shape=(4,))
    rh = CartesianRestrictionOperator(mesh)
    rf = rh(_Func1D(c, _x))
    for i in range(4):
        assert sympy.simplify(rf((i,)) - c) == 0  # type: ignore[arg-type]


def test_cell_average_constant_1d() -> None:
    assert_cell_average_constant_1d()


def assert_cell_average_linear_1d() -> None:
    """Cell average of f=x over [a, a+h] is a + h/2 (cell midpoint)."""
    a = sympy.Symbol("a")
    mesh = CartesianMesh(origin=(a,), spacing=(_H,), shape=(3,))
    rh = CartesianRestrictionOperator(mesh)
    rf = rh(_Func1D(_x, _x))
    for i in range(3):
        expected = a + (_H * i + _H / 2)
        assert sympy.simplify(rf((i,)) - expected) == 0  # type: ignore[arg-type]


def test_cell_average_linear_1d() -> None:
    assert_cell_average_linear_1d()


def assert_cell_average_quadratic_1d() -> None:
    """Cell average of f=x² over [i*h, (i+1)*h] from origin=0."""
    mesh = CartesianMesh(origin=(_ZERO,), spacing=(_H,), shape=(2,))
    rh = CartesianRestrictionOperator(mesh)
    rf = rh(_Func1D(_x**2, _x))
    # ∫_{i*h}^{(i+1)*h} x² dx / h = h²(i² + i + 1/3)
    for i in range(2):
        expected = _H**2 * (i**2 + i + sympy.Rational(1, 3))
        assert sympy.simplify(rf((i,)) - expected) == 0  # type: ignore[arg-type]


def test_cell_average_quadratic_1d() -> None:
    assert_cell_average_quadratic_1d()


def assert_cell_average_separable_2d() -> None:
    """Cell average of f=x*y factors as (avg x)(avg y)."""
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


def test_cell_average_separable_2d() -> None:
    assert_cell_average_separable_2d()


def assert_restriction_mesh_attribute() -> None:
    """Rₕ.mesh and (Rₕ f).mesh both point to the same CartesianMesh."""
    mesh = CartesianMesh(origin=(_ZERO,), spacing=(_H,), shape=(2,))
    rh = CartesianRestrictionOperator(mesh)
    rf = rh(_Func1D(sympy.Integer(1), _x))
    assert rh.mesh is mesh
    assert rf.mesh is mesh


def test_restriction_mesh_attribute() -> None:
    assert_restriction_mesh_attribute()
