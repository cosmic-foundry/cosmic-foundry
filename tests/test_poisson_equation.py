"""Lane C: symbolic verification of PoissonEquation's divergence form.

Verifies that the divergence-theorem form of PoissonEquation,
    ∮_∂Ω (-∇φ)·n̂ dA = ∫_Ω ρ dV,
recovers -∇²φ = ρ under the Cartesian chart via integration by parts.
"""

from __future__ import annotations

import sympy

from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.continuous.field import Field
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.continuous.poisson_equation import PoissonEquation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_x, _y = sympy.symbols("x y")
_PI = sympy.pi


class _ScalarField(Field):
    """Concrete scalar Field wrapping a SymPy expression."""

    def __init__(
        self,
        manifold: Manifold,
        expr: sympy.Expr,
        symbols: tuple[sympy.Symbol, ...],
    ) -> None:
        self._manifold = manifold
        self._expr = expr
        self._symbols = symbols

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def expr(self) -> sympy.Expr:
        return self._expr

    @property
    def symbols(self) -> tuple[sympy.Symbol, ...]:
        return self._symbols


class _ConcretePoissonEquation(PoissonEquation):
    """Minimal concrete PoissonEquation on a given manifold with given source."""

    def __init__(self, manifold: Manifold, source: Field) -> None:
        self._manifold = manifold
        self._source = source

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def source(self) -> Field:
        return self._source


# Manufactured solution: φ = sin(πx)sin(πy), ρ = 2π²sin(πx)sin(πy)
_PHI_EXPR = sympy.sin(_PI * _x) * sympy.sin(_PI * _y)
_RHO_EXPR = 2 * _PI**2 * sympy.sin(_PI * _x) * sympy.sin(_PI * _y)

# ---------------------------------------------------------------------------
# Lane C assertions
# ---------------------------------------------------------------------------


def assert_operator_recovers_negated_laplacian() -> None:
    """Lane C: L(φ) = -∇²φ equals the manufactured source ρ.

    Verifies that PoissonEquation.__call__(φ) returns the negated Laplacian
    -∇²φ and that this matches the source ρ for the manufactured pair
    φ = sin(πx)sin(πy), ρ = 2π²sin(πx)sin(πy).
    """
    manifold = EuclideanManifold(2)
    phi = _ScalarField(manifold, _PHI_EXPR, (manifold.symbols[0], manifold.symbols[1]))
    source = _ScalarField(
        manifold, _RHO_EXPR, (manifold.symbols[0], manifold.symbols[1])
    )
    eq = _ConcretePoissonEquation(manifold, source)

    result = eq(phi)
    x, y = manifold.symbols
    assert sympy.simplify(result.expr.subs([(x, _x), (y, _y)]) - _RHO_EXPR) == 0


def test_operator_recovers_negated_laplacian() -> None:
    assert_operator_recovers_negated_laplacian()


def assert_flux_divergence_agrees_with_operator() -> None:
    """Lane C: ∇·(flux(φ)) = L(φ); the flux route and operator route agree.

    Verifies that applying the divergence to eq.flux(φ) = -∇φ gives the
    same result as eq(φ) = -∇²φ.  This is the chain

        ∮_∂Ω F(φ)·n̂ dA → ∫_Ω ∇·F(φ) dV = ∫_Ω (-∇²φ) dV = ∫_Ω ρ dV,

    confirming the divergence-theorem form recovers the strong-form PDE.
    """
    manifold = EuclideanManifold(2)
    x, y = manifold.symbols
    phi_expr = _PHI_EXPR.subs([(_x, x), (_y, y)])
    rho_expr = _RHO_EXPR.subs([(_x, x), (_y, y)])

    phi = _ScalarField(manifold, phi_expr, manifold.symbols)
    source = _ScalarField(manifold, rho_expr, manifold.symbols)
    eq = _ConcretePoissonEquation(manifold, source)

    # Route 1: divergence of flux
    neg_grad = eq.flux(phi)
    div_flux: sympy.Expr = sum(  # type: ignore[assignment]
        sympy.diff(neg_grad.component(i), manifold.symbols[i])
        for i in range(manifold.ndim)
    )

    # Route 2: direct operator application
    lph = eq(phi).expr

    assert sympy.simplify(div_flux - lph) == 0
    assert sympy.simplify(div_flux - eq.source.expr) == 0


def test_flux_divergence_agrees_with_operator() -> None:
    assert_flux_divergence_agrees_with_operator()
