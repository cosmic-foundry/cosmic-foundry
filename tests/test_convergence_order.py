"""Parametric convergence order verification for all registered convergent classes.

Every concrete subclass of a convergent ABC must have instances registered in
tests/support/oracles/__init__.py.  conftest.py enforces this at collection
time and parametrizes this test over every registered instance.

Exact values are computed automatically: given instance.continuous_operator L
and a test field phi, the exact discrete output is

    Rₕ(L phi)

where the restriction degree is ndim - (degree of L phi).  No oracle `error`
method is needed; the test is fully self-contained given `continuous_operator`
and `CartesianRestrictionOperator`.
"""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianRestrictionOperator,
)
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    ZeroForm,
)
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator


class _ZeroFormField(ZeroForm[Any]):
    def __init__(
        self,
        manifold: Any,
        expr: sympy.Expr,
        symbols: tuple[sympy.Symbol, ...],
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


def test_convergence_order(convergence_case: DiscreteOperator) -> None:
    instance = convergence_case
    h = sympy.Symbol("h", positive=True)
    order = instance.order
    n = order // 2
    num_cells = 2 * n + 2

    space = EuclideanManifold(1)
    x = space.atlas[0].symbols[0]
    mesh = CartesianMesh(
        origin=(sympy.Integer(0),),
        spacing=(h,),
        shape=(num_cells,),
    )
    ndim = len(mesh._shape)

    coeffs = sympy.symbols(f"a:{order + 4}")
    phi_expr: sympy.Expr = sum(c * x**k for k, c in enumerate(coeffs))
    phi = _ZeroFormField(space, phi_expr, (x,))

    Rh_cell = CartesianRestrictionOperator(mesh, degree=ndim)
    U = Rh_cell(phi)
    numerical_mf = instance(U)

    cont_result = instance.continuous_operator(phi)
    assert isinstance(cont_result, DifferentialForm)
    restriction_degree = ndim - cont_result.degree
    Rh_exact = CartesianRestrictionOperator(mesh, degree=restriction_degree)
    exact_mf = Rh_exact(cont_result)

    if restriction_degree < ndim:
        test_idx: Any = (0, (n,))
    else:
        test_idx = (n,)

    numerical = numerical_mf(test_idx)
    exact = exact_mf(test_idx)
    error = sympy.expand(sympy.simplify(numerical - exact))
    poly = sympy.Poly(error, h)
    for k in range(order):
        assert poly.nth(k) == 0, (
            f"Unexpected O(h^{k}) term in {type(instance).__name__}"
            f"(order={order}): {poly.nth(k)}"
        )
    assert poly.nth(order) != 0, (
        f"Missing O(h^{order}) leading term in "
        f"{type(instance).__name__}(order={order})"
    )
