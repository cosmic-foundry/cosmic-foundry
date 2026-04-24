"""Lane C verification for DiffusiveFlux(2) and DiffusiveFlux(4).

Two assert_* functions verify the falsifiable claim: DiffusiveFlux(order)
applied to exact SymPy cell averages of a generic polynomial produces a
face flux whose error is O(hᵖ), where p = order.

The constructor derives its own stencil coefficients from the cell-average
moment system, so reconstruction and face quadrature theory are not tested
separately here — they are the derivation, not the result.  These composite
tests verify that the derived coefficients are correctly applied in __call__
against a CartesianMesh.

Validity conditions (uniform 1D Cartesian grid, smooth data, no limiters):

* Cell averages are computed analytically by CartesianRestrictionOperator.
* The test polynomial is in C^∞; all derivatives exist to any order.
* Face is interior; boundary effects are absent.
"""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianRestrictionOperator,
)
from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.continuous.field import Field

_a0, _a1, _a2, _a3, _a4, _a5, _a6 = sympy.symbols("a:7")


class _TestField(Field[Any, sympy.Expr]):
    """Minimal concrete Field for Lane C composite tests."""

    def __init__(
        self, manifold: Any, expr: sympy.Expr, symbols: tuple[sympy.Symbol, ...]
    ) -> None:
        self._manifold = manifold
        self._expr = expr
        self._symbols = symbols

    @property
    def manifold(self) -> Any:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._symbols


def assert_composite_order_2() -> None:
    """DiffusiveFlux(2) approximates the exact face-averaged normal flux to O(h²).

    Uses a 1D CartesianMesh with symbolic spacing h_val and
    CartesianRestrictionOperator to produce exact SymPy cell averages of a
    degree-5 polynomial.  DiffusiveFlux(2).__call__ is applied at an interior
    face; the error against the exact -φ'(x_face) is verified to be O(h²).
    """
    h_val = sympy.Symbol("h_val", positive=True)
    mesh = CartesianMesh(
        origin=(sympy.Integer(0),),
        spacing=(h_val,),
        shape=(4,),
    )
    space = EuclideanManifold(1)
    x_sym = space.atlas[0].symbols[0]
    phi_expr = (
        _a0
        + _a1 * x_sym
        + _a2 * x_sym**2
        + _a3 * x_sym**3
        + _a4 * x_sym**4
        + _a5 * x_sym**5
    )
    U = CartesianRestrictionOperator(mesh)(_TestField(space, phi_expr, (x_sym,)))

    numerical = DiffusiveFlux(2)(U, mesh, 0, (1,))
    x_face = 2 * h_val
    exact = -sympy.diff(phi_expr, x_sym).subs(x_sym, x_face) * mesh.face_area(0)

    error = sympy.expand(sympy.simplify(numerical - exact))
    assert sympy.Poly(error, h_val).nth(0) == 0
    assert sympy.Poly(error, h_val).nth(1) == 0
    assert sympy.Poly(error, h_val).nth(2) != 0


def assert_composite_order_4() -> None:
    """DiffusiveFlux(4) approximates the exact face-averaged normal flux to O(h⁴).

    Uses a 1D CartesianMesh with 6 cells so the 4-cell stencil has room at
    the chosen interior face (between cells 2 and 3).  The degree-6 polynomial
    ensures the O(h⁴) leading term is present.
    """
    h_val = sympy.Symbol("h_val", positive=True)
    mesh = CartesianMesh(
        origin=(sympy.Integer(0),),
        spacing=(h_val,),
        shape=(6,),
    )
    space = EuclideanManifold(1)
    x_sym = space.atlas[0].symbols[0]
    phi_expr = (
        _a0
        + _a1 * x_sym
        + _a2 * x_sym**2
        + _a3 * x_sym**3
        + _a4 * x_sym**4
        + _a5 * x_sym**5
        + _a6 * x_sym**6
    )
    U = CartesianRestrictionOperator(mesh)(_TestField(space, phi_expr, (x_sym,)))

    numerical = DiffusiveFlux(4)(U, mesh, 0, (2,))
    x_face = 3 * h_val
    exact = -sympy.diff(phi_expr, x_sym).subs(x_sym, x_face) * mesh.face_area(0)

    error = sympy.expand(sympy.simplify(numerical - exact))
    for p in range(4):
        assert sympy.Poly(error, h_val).nth(p) == 0, (
            f"Unexpected O(h^{p}) term in composite p=4 flux error: "
            f"{sympy.Poly(error, h_val).nth(p)}"
        )
    assert sympy.Poly(error, h_val).nth(4) != 0


def test_composite_order_2() -> None:
    assert_composite_order_2()


def test_composite_order_4() -> None:
    assert_composite_order_4()
