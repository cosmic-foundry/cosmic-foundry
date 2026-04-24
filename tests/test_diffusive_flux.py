"""Lane C symbolic derivations for DiffusiveFlux(2) and DiffusiveFlux(4).

Eight assert_* functions, one per component per order:

    Reconstruction    p=2  assert_reconstruction_order_2
    Face quadrature   p=2  assert_face_quadrature_order_2
    Deconvolution     p=2  assert_deconvolution_order_2
    Composite flux    p=2  assert_composite_order_2
    Reconstruction    p=4  assert_reconstruction_order_4
    Face quadrature   p=4  assert_face_quadrature_order_4
    Deconvolution     p=4  assert_deconvolution_order_4
    Composite flux    p=4  assert_composite_order_4

Each function expands the approximation error as a power series in h using a
generic degree-6 polynomial test function and verifies that the leading
non-zero term is O(hᵖ) — no lower-order terms are present.

The test functions are thin wrappers that call the assert_* functions and
nothing else.  Notebooks import the assert_* functions and display their
source via inspect.getsource so that readers see exactly what CI runs.

Validity conditions (uniform 1D Cartesian grid, uniform spacing h):

* Smooth initial data: the test polynomial is in C^∞.
* Face quadrature tests use a 1D integral along the face direction (transverse
  to the normal); in 1D the face is a point and quadrature is trivially exact,
  so the tests work with a 1D slice and verify the abstract quadrature rule
  (midpoint p=2, Simpson p=4) against the exact face average.
* No limiters or non-conservative terms are present.
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


class _TestField(Field[Any, sympy.Expr]):
    """Minimal concrete SymbolicFunction for use in Lane C composite tests."""

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


# ---------------------------------------------------------------------------
# Shared symbols and helpers
# ---------------------------------------------------------------------------

_h, _x, _y = sympy.symbols("h x y", positive=True)
_a0, _a1, _a2, _a3, _a4, _a5, _a6 = sympy.symbols("a:7")

# Degree-6 test polynomial (generic smooth function, all coefficients free)
_PHI = (
    _a0 + _a1 * _x + _a2 * _x**2 + _a3 * _x**3 + _a4 * _x**4 + _a5 * _x**5 + _a6 * _x**6
)


def _cell_average(expr: sympy.Expr, center: sympy.Expr) -> sympy.Expr:
    """Exact cell average of expr over [center - h/2, center + h/2]."""
    lo = center - _h / 2
    hi = center + _h / 2
    return sympy.integrate(expr, (_x, lo, hi)) / _h


def _error_order(error: sympy.Expr) -> int:
    """Return the lowest power of h present in error (expanded in h)."""
    expanded = sympy.expand(error)
    for p in range(7):
        coeff = sympy.Poly(expanded, _h).nth(p) if expanded != 0 else 0
        if coeff != 0:
            return p
    return 7  # all terms through h^6 vanish


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def assert_reconstruction_order_2() -> None:
    """Centered difference of cell averages approximates φ'(face) to O(h²).

    The face is at x=0.  Cell i has center -h/2; cell i+1 has center +h/2.

        (φ̄_{i+1} - φ̄_i) / h  =  φ'(0) + (h²/12)φ'''(0) + O(h⁴)

    Taylor expansion confirms the leading error is h² (coefficient h²/12·a3).
    """
    phi_bar_L = _cell_average(_PHI, -_h / 2)
    phi_bar_R = _cell_average(_PHI, +_h / 2)
    approx = (phi_bar_R - phi_bar_L) / _h
    exact = sympy.diff(_PHI, _x).subs(_x, 0)
    error = sympy.expand(approx - exact)
    # No constant or h^1 terms
    assert sympy.Poly(error, _h).nth(0) == 0
    assert sympy.Poly(error, _h).nth(1) == 0
    # Leading error is h^2
    assert sympy.Poly(error, _h).nth(2) != 0


def assert_reconstruction_order_4() -> None:
    """Four-point antisymmetric cell-average stencil approximates φ'(face) to O(h⁴).

    Stencil derived by enforcing antisymmetry and vanishing error through h³:

        (φ̄_{i-1} - 15φ̄_i + 15φ̄_{i+1} - φ̄_{i+2}) / (12h)  =  φ'(0) + O(h⁴)

    Coefficients 1/12, -15/12, 15/12, -1/12 are the unique antisymmetric
    solution to Σ c_j ξ_j = 1, Σ c_j (ξ_j³/6 + ξ_j h²/24) = 0 with
    ξ ∈ {-3h/2, -h/2, h/2, 3h/2}.
    """
    phi_bar_m1 = _cell_average(_PHI, -3 * _h / 2)
    phi_bar_0 = _cell_average(_PHI, -_h / 2)
    phi_bar_p1 = _cell_average(_PHI, +_h / 2)
    phi_bar_p2 = _cell_average(_PHI, +3 * _h / 2)
    approx = (phi_bar_m1 - 15 * phi_bar_0 + 15 * phi_bar_p1 - phi_bar_p2) / (12 * _h)
    exact = sympy.diff(_PHI, _x).subs(_x, 0)
    error = sympy.expand(approx - exact)
    for p in range(4):
        assert sympy.Poly(error, _h).nth(p) == 0, (
            f"Unexpected O(h^{p}) term in reconstruction error: "
            f"{sympy.Poly(error, _h).nth(p)}"
        )
    assert sympy.Poly(error, _h).nth(4) != 0


# ---------------------------------------------------------------------------
# Face quadrature
# ---------------------------------------------------------------------------


def assert_face_quadrature_order_2() -> None:
    """Midpoint rule approximates the face average of a smooth function to O(h²).

    The face is a 1D interval [0, h] in the transverse direction y.
    Midpoint rule: f(h/2).  Exact: (1/h) ∫_0^h f(y) dy.

        f(h/2)  =  (1/h) ∫_0^h f(y) dy  +  O(h²)

    This is the abstract accuracy claim for the face quadrature component of
    DiffusiveFlux(2).  In 1D the face is a point and quadrature is exact;
    the error O(h²) is only relevant in 2D+ where the face has non-zero extent.
    """
    f = _a0 + _a1 * _y + _a2 * _y**2 + _a3 * _y**3 + _a4 * _y**4
    exact_avg = sympy.integrate(f, (_y, 0, _h)) / _h
    midpoint = f.subs(_y, _h / 2)
    error = sympy.expand(midpoint - exact_avg)
    assert sympy.Poly(error, _h).nth(0) == 0
    assert sympy.Poly(error, _h).nth(1) == 0
    assert sympy.Poly(error, _h).nth(2) != 0


def assert_face_quadrature_order_4() -> None:
    """Simpson's rule approximates the face average of a smooth function to O(h⁴).

    The face is a 1D interval [0, h].  Simpson's rule:
        (1/6)(f(0) + 4f(h/2) + f(h))  =  (1/h) ∫_0^h f(y) dy  +  O(h⁴)

    This is the abstract accuracy claim for the face quadrature component of
    DiffusiveFlux(4) in 2D+.
    """
    f = _a0 + _a1 * _y + _a2 * _y**2 + _a3 * _y**3 + _a4 * _y**4 + _a5 * _y**5
    exact_avg = sympy.integrate(f, (_y, 0, _h)) / _h
    simpson = (f.subs(_y, 0) + 4 * f.subs(_y, _h / 2) + f.subs(_y, _h)) / 6
    error = sympy.expand(simpson - exact_avg)
    for p in range(4):
        assert sympy.Poly(error, _h).nth(p) == 0, (
            f"Unexpected O(h^{p}) term in Simpson quadrature error: "
            f"{sympy.Poly(error, _h).nth(p)}"
        )
    assert sympy.Poly(error, _h).nth(4) != 0


# ---------------------------------------------------------------------------
# Deconvolution
# ---------------------------------------------------------------------------


def assert_deconvolution_order_2() -> None:
    """Cell average equals cell-center point value up to O(h²).

    The identity operator (φ̄_i used directly as φ(x_i)) satisfies:

        φ̄_i  =  φ(x_i) + (h²/24)φ''(x_i) + O(h⁴)

    so the error of using φ̄_i as the point value is O(h²).  For p=2,
    the deconvolution is the identity; the O(h²) residual does not degrade
    the overall O(h²) accuracy of the scheme.
    """
    center = sympy.Symbol("c")
    phi_at_center = _PHI.subs(_x, center)
    phi_bar = sympy.integrate(_PHI, (_x, center - _h / 2, center + _h / 2)) / _h
    # Deconvolution p=2: identity — use φ̄ as the point value
    error = sympy.expand(sympy.simplify(phi_bar - phi_at_center))
    # expand around center=0 for coefficient extraction
    error_at_0 = error.subs(center, 0)
    assert sympy.Poly(error_at_0, _h).nth(0) == 0
    assert sympy.Poly(error_at_0, _h).nth(1) == 0
    assert sympy.Poly(error_at_0, _h).nth(2) != 0


def assert_deconvolution_order_4() -> None:
    """Corrected cell average recovers the point value up to O(h⁴).

    The deconvolution formula

        φ(x_i)  =  φ̄_i - (φ̄_{i-1} - 2φ̄_i + φ̄_{i+1}) / 24  +  O(h⁴)

    follows from inverting φ̄_i = φ(x_i) + (h²/24)φ''(x_i) + O(h⁴) and
    approximating φ''(x_i) ≈ (φ̄_{i-1} - 2φ̄_i + φ̄_{i+1}) / h².
    The resulting error is O(h⁴).
    """
    phi_bar_m1 = _cell_average(_PHI, -_h)
    phi_bar_0 = _cell_average(_PHI, sympy.Integer(0))
    phi_bar_p1 = _cell_average(_PHI, +_h)
    deconvolved = phi_bar_0 - (phi_bar_m1 - 2 * phi_bar_0 + phi_bar_p1) / 24
    exact = _PHI.subs(_x, 0)
    error = sympy.expand(deconvolved - exact)
    for p in range(4):
        assert sympy.Poly(error, _h).nth(p) == 0, (
            f"Unexpected O(h^{p}) term in deconvolution error: "
            f"{sympy.Poly(error, _h).nth(p)}"
        )
    assert sympy.Poly(error, _h).nth(4) != 0


# ---------------------------------------------------------------------------
# Composite face flux
# ---------------------------------------------------------------------------


def assert_composite_order_2() -> None:
    """DiffusiveFlux(2) approximates the exact face-averaged normal flux to O(h²).

    The exact face-averaged normal flux at x=0 for F = -∇φ is -φ'(0).
    DiffusiveFlux(2) returns -(φ̄_{i+1} - φ̄_i)/h · face_area.

    This test uses CartesianMesh(1D, N=4, h=h) and CartesianRestrictionOperator
    to produce exact SymPy cell averages of a known test field, then calls
    DiffusiveFlux(2) and verifies the error is O(h²).
    """
    h_val = sympy.Symbol("h_val", positive=True)
    mesh = CartesianMesh(
        origin=(sympy.Integer(0),),
        spacing=(h_val,),
        shape=(4,),
    )
    space = EuclideanManifold(1)
    chart = space.atlas[0]
    x_sym = chart.symbols[0]
    phi_expr = (
        _a0
        + _a1 * x_sym
        + _a2 * x_sym**2
        + _a3 * x_sym**3
        + _a4 * x_sym**4
        + _a5 * x_sym**5
    )
    test_field = _TestField(space, phi_expr, (x_sym,))
    Rh = CartesianRestrictionOperator(mesh)
    U = Rh(test_field)

    flux = DiffusiveFlux(2)
    # Face between cells 1 and 2 (interior face at x = 2h)
    numerical = flux(U, mesh, 0, (1,))

    # Exact face-averaged normal flux: -φ'(x_face) · face_area
    # Face is at x = origin + (idx+1)*spacing = 0 + 2*h_val = 2*h_val
    x_face = 2 * h_val
    exact = -sympy.diff(phi_expr, x_sym).subs(x_sym, x_face) * mesh.face_area(0)

    error = sympy.expand(sympy.simplify(numerical - exact))
    assert sympy.Poly(error, h_val).nth(0) == 0
    assert sympy.Poly(error, h_val).nth(1) == 0
    assert sympy.Poly(error, h_val).nth(2) != 0


def assert_composite_order_4() -> None:
    """DiffusiveFlux(4) approximates the exact face-averaged normal flux to O(h⁴).

    Uses a 1D CartesianMesh with 6 cells so that the 4-wide stencil has room
    at the chosen interior face.  The exact face flux is -φ'(x_face).
    DiffusiveFlux(4) returns the 4-point stencil result; the error is O(h⁴).
    """
    h_val = sympy.Symbol("h_val", positive=True)
    mesh = CartesianMesh(
        origin=(sympy.Integer(0),),
        spacing=(h_val,),
        shape=(6,),
    )
    space = EuclideanManifold(1)
    chart = space.atlas[0]
    x_sym = chart.symbols[0]
    phi_expr = (
        _a0
        + _a1 * x_sym
        + _a2 * x_sym**2
        + _a3 * x_sym**3
        + _a4 * x_sym**4
        + _a5 * x_sym**5
        + _a6 * x_sym**6
    )
    test_field = _TestField(space, phi_expr, (x_sym,))
    Rh = CartesianRestrictionOperator(mesh)
    U = Rh(test_field)

    flux = DiffusiveFlux(4)
    # Face between cells 2 and 3 (interior, stencil uses cells 1,2,3,4)
    numerical = flux(U, mesh, 0, (2,))

    x_face = 3 * h_val  # origin + (2+1)*spacing
    exact = -sympy.diff(phi_expr, x_sym).subs(x_sym, x_face) * mesh.face_area(0)

    error = sympy.expand(sympy.simplify(numerical - exact))
    for p in range(4):
        assert sympy.Poly(error, h_val).nth(p) == 0, (
            f"Unexpected O(h^{p}) term in composite p=4 flux error: "
            f"{sympy.Poly(error, h_val).nth(p)}"
        )
    assert sympy.Poly(error, h_val).nth(4) != 0


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

DIFFUSIVE_FLUX_CASES = [
    ("reconstruction_2", assert_reconstruction_order_2),
    ("face_quadrature_2", assert_face_quadrature_order_2),
    ("deconvolution_2", assert_deconvolution_order_2),
    ("composite_2", assert_composite_order_2),
    ("reconstruction_4", assert_reconstruction_order_4),
    ("face_quadrature_4", assert_face_quadrature_order_4),
    ("deconvolution_4", assert_deconvolution_order_4),
    ("composite_4", assert_composite_order_4),
]


def test_reconstruction_order_2() -> None:
    assert_reconstruction_order_2()


def test_face_quadrature_order_2() -> None:
    assert_face_quadrature_order_2()


def test_deconvolution_order_2() -> None:
    assert_deconvolution_order_2()


def test_composite_order_2() -> None:
    assert_composite_order_2()


def test_reconstruction_order_4() -> None:
    assert_reconstruction_order_4()


def test_face_quadrature_order_4() -> None:
    assert_face_quadrature_order_4()


def test_deconvolution_order_4() -> None:
    assert_deconvolution_order_4()


def test_composite_order_4() -> None:
    assert_composite_order_4()
