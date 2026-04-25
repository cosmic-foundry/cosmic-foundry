"""ConvergenceOracle for _AssembledFVMOperator.

Tests the commutation diagram ‖Lₕ Rₕ f − Rₕ L f‖_{∞,h} = O(hᵖ) at order p
for FVMDiscretization assembled operators.

Manufactured solution: a polynomial of degree order+3 on a 1D CartesianMesh
with symbolic spacing h.  The same stencil layout as DiffusiveFluxOracle is
used so the test cell is strictly interior.

Stencil layout for order p, n = p//2:
    cells:   0 ... n-1 | n ... 2n+1
    test cell: n   (idx = (n,))
    width:   num_cells = 2*n+2
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
from cosmic_foundry.geometry.fvm_discretization import (
    FVMDiscretization,
    _AssembledFVMOperator,
)
from cosmic_foundry.theory.continuous.differential_form import ZeroForm
from cosmic_foundry.theory.continuous.diffusion_operator import DiffusionOperator
from cosmic_foundry.theory.continuous.field import Field
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.continuous.poisson_equation import PoissonEquation


class _Field(Field[Any, sympy.Expr]):
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


class _ConcretePoissonEquation(PoissonEquation):
    """Minimal concrete PoissonEquation for oracle use."""

    def __init__(self, manifold: Manifold, source: ZeroForm) -> None:
        self._manifold = manifold
        self._source = source

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def source(self) -> ZeroForm:
        return self._source


class FVMDiscretizationOracle:
    def instances(self) -> list[_AssembledFVMOperator]:
        manifold = EuclideanManifold(1)
        diffusion_op = DiffusionOperator(manifold)
        x = manifold.atlas[0].symbols[0]
        zero_source = _Field(manifold, sympy.Integer(0), (x,))
        poisson = _ConcretePoissonEquation(manifold, zero_source)  # type: ignore[arg-type]
        dummy_mesh = CartesianMesh(
            origin=(sympy.Integer(0),),
            spacing=(sympy.Integer(1),),
            shape=(4,),
        )
        lo = DiffusiveFlux.min_order
        step = DiffusiveFlux.order_step
        return [
            FVMDiscretization(dummy_mesh, DiffusiveFlux(lo, diffusion_op))(poisson),
            FVMDiscretization(dummy_mesh, DiffusiveFlux(lo + step, diffusion_op))(
                poisson
            ),
        ]

    def error(self, instance: _AssembledFVMOperator, h: sympy.Symbol) -> sympy.Expr:
        order = instance.order
        n = order // 2
        num_cells = 2 * n + 2
        idx_test = (n,)
        mesh = CartesianMesh(
            origin=(sympy.Integer(0),),
            spacing=(h,),
            shape=(num_cells,),
        )
        space = EuclideanManifold(1)
        x = space.atlas[0].symbols[0]
        degree = order + 3
        coeffs = sympy.symbols(f"a:{degree + 1}")
        phi = sum(c * x**k for k, c in enumerate(coeffs))
        phi_field = _Field(space, phi, (x,))
        U = CartesianRestrictionOperator(mesh)(phi_field)
        numerical = instance(U)(idx_test)
        neg_laplacian = -sympy.diff(phi, x, 2)
        exact_field = _Field(space, neg_laplacian, (x,))
        exact = CartesianRestrictionOperator(mesh)(exact_field)(idx_test)  # type: ignore[arg-type]
        return sympy.expand(sympy.simplify(numerical - exact))


__all__ = ["FVMDiscretizationOracle"]
