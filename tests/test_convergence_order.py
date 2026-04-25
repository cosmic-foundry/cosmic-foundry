"""Convergence order verification for all concrete DiscreteOperator subclasses.

When adding a new concrete DiscreteOperator subclass, add its instances to
_INSTANCES below.  Each instance must carry `order` and `continuous_operator`;
the test auto-computes the exact value via Rₕ(L φ) and verifies the error
polynomial has zeros at h⁰…h^{p-1} and a nonzero h^p leading term.
"""

from __future__ import annotations

from typing import Any

import pytest
import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianRestrictionOperator,
)
from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.geometry.fvm_discretization import FVMDiscretization
from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    ZeroForm,
)


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


_manifold = EuclideanManifold(1)
_dummy_mesh = CartesianMesh(
    origin=(sympy.Integer(0),), spacing=(sympy.Integer(1),), shape=(4,)
)

_INSTANCES = [
    DiffusiveFlux(DiffusiveFlux.min_order, _manifold),
    DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold),
    FVMDiscretization(_dummy_mesh, DiffusiveFlux(DiffusiveFlux.min_order, _manifold))(),
    FVMDiscretization(
        _dummy_mesh,
        DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold),
    )(),
]


@pytest.mark.parametrize(
    "instance",
    _INSTANCES,
    ids=[f"{type(i).__name__}(order={i.order})" for i in _INSTANCES],
)
def test_convergence_order(instance: Any) -> None:
    h = sympy.Symbol("h", positive=True)
    order = instance.order
    n = order // 2

    space = EuclideanManifold(1)
    x = space.atlas[0].symbols[0]
    mesh = CartesianMesh(
        origin=(sympy.Integer(0),),
        spacing=(h,),
        shape=(2 * n + 2,),
    )
    ndim = len(mesh._shape)

    coeffs = sympy.symbols(f"a:{order + 4}")
    phi_expr: sympy.Expr = sum(c * x**k for k, c in enumerate(coeffs))
    phi = _ZeroFormField(space, phi_expr, (x,))

    U = CartesianRestrictionOperator(mesh, degree=ndim)(phi)
    numerical_mf = instance(U)

    cont_result = instance.continuous_operator(phi)
    assert isinstance(cont_result, DifferentialForm)
    restriction_degree = ndim - cont_result.degree
    Rh_exact = CartesianRestrictionOperator(mesh, degree=restriction_degree)
    exact_mf = Rh_exact(cont_result)

    test_idx: Any = (0, (n,)) if restriction_degree < ndim else (n,)
    error = sympy.expand(sympy.simplify(numerical_mf(test_idx) - exact_mf(test_idx)))
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
