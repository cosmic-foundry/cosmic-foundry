"""CartesianRestrictionOperator: analytic cell-average restriction on CartesianMesh."""

from __future__ import annotations

from itertools import product
from typing import Any

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.discrete.restriction_operator import RestrictionOperator
from cosmic_foundry.theory.foundation.symbolic_function import SymbolicFunction


class _CartesianCellAverage(MeshFunction[sympy.Expr]):
    """Cell-averaged values on a CartesianMesh."""

    def __init__(
        self,
        mesh: CartesianMesh,
        values: dict[tuple[int, ...], sympy.Expr],
    ) -> None:
        self._mesh = mesh
        self._values = values

    @property
    def mesh(self) -> CartesianMesh:
        return self._mesh

    def __call__(self, idx: tuple[int, ...]) -> sympy.Expr:
        return self._values[idx]


class CartesianRestrictionOperator(RestrictionOperator[Any, sympy.Expr]):
    """Restriction operator Rₕ for CartesianMesh via analytic SymPy integration.

    (Rₕ f)ᵢ = |Ωᵢ|⁻¹ ∫_Ωᵢ f dV

    The integral is computed analytically by integrating the SymPy expression
    of a SymbolicFunction over each cell interval along each axis in turn.
    The result is a _CartesianCellAverage whose values are exact SymPy Exprs.

    f must be a SymbolicFunction with one symbol per mesh dimension, in the
    same axis order as CartesianMesh.coordinate.
    """

    def __init__(self, mesh: CartesianMesh) -> None:
        self._mesh = mesh

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, f: SymbolicFunction) -> MeshFunction[sympy.Expr]:
        mesh = self._mesh
        values: dict[tuple[int, ...], sympy.Expr] = {}
        for idx in product(*[range(s) for s in mesh._shape]):
            expr = f.expr
            for i, sym in enumerate(f.symbols):
                lo = mesh._origin[i] + sympy.Integer(idx[i]) * mesh._spacing[i]
                hi = lo + mesh._spacing[i]
                expr = sympy.integrate(expr, (sym, lo, hi))
            values[idx] = sympy.simplify(expr / mesh.cell_volume)
        return _CartesianCellAverage(mesh, values)


__all__ = ["CartesianRestrictionOperator"]
