"""ConvergenceOracle for DiffusiveFlux.

Manufactured solution: a polynomial of degree order+3 on a 1D CartesianMesh
with symbolic spacing h.  Cell averages are computed analytically via
CartesianRestrictionOperator.  The interior face sits symmetrically in the
stencil so no boundary cells are touched.

Stencil layout for order p, n = p//2:
    cells:   0 ... n-1 | n ... 2n+1
    face:    between cells n and n+1   (idx_low = (n,))
    x_face:  (n+1)*h
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
from cosmic_foundry.theory.continuous.field import Field


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


class DiffusiveFluxOracle:
    def instances(self) -> list[DiffusiveFlux]:
        lo = DiffusiveFlux.min_order
        step = DiffusiveFlux.order_step
        return [DiffusiveFlux(lo), DiffusiveFlux(lo + step)]

    def error(self, instance: DiffusiveFlux, h: sympy.Symbol) -> sympy.Expr:
        order = instance.order
        n = order // 2
        num_cells = 2 * n + 2
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
        U = CartesianRestrictionOperator(mesh)(_Field(space, phi, (x,)))
        idx_low = (n,)
        face_fluxes = instance(U)
        numerical = face_fluxes((0, idx_low))
        x_face = (n + 1) * h
        exact = -sympy.diff(phi, x).subs(x, x_face) * mesh.face_area(0)
        return sympy.expand(sympy.simplify(numerical - exact))


__all__ = ["DiffusiveFluxOracle"]
