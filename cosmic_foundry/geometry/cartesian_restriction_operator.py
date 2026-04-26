"""CartesianRestrictionOperator: analytic restriction on CartesianMesh."""

from __future__ import annotations

from itertools import product
from typing import Any

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_form import OneForm
from cosmic_foundry.theory.continuous.symbolic_function import SymbolicFunction
from cosmic_foundry.theory.discrete.cell_field import CellField
from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.face_field import FaceField, _CallableFaceField
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.restriction_operator import RestrictionOperator


class _CartesianCellAverage(CellField[sympy.Expr]):
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

    def __call__(self, idx: tuple[int, ...]) -> sympy.Expr:  # type: ignore[override]
        return self._values[idx]


class CartesianRestrictionOperator(RestrictionOperator[Any, sympy.Expr]):
    """Restriction operator Rₕ for CartesianMesh via analytic SymPy integration.

    degree == ndim (default): cell-average restriction
        (Rₕ f)ᵢ = |Ωᵢ|⁻¹ ∫_Ωᵢ f dV

    degree == ndim - 1: face-normal restriction of a OneForm F
        (Rₕ F)_{a,i} = ∫_{face (a,i)} F·ê_a dA
                     = ∫_{transverse dims} F.component(a)|_{x_a = face_x} dx_⊥

    In both cases the integral is computed analytically via SymPy.
    """

    def __init__(self, mesh: CartesianMesh, degree: int | None = None) -> None:
        self._mesh = mesh
        ndim = len(mesh._shape)
        self._degree = ndim if degree is None else degree

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def degree(self) -> int:
        return self._degree

    def __call__(self, f: SymbolicFunction) -> DiscreteField[sympy.Expr]:  # type: ignore[override]  # LSP: RestrictionOperator.__call__ takes (M, V) not SymbolicFunction; deferred to a later PR
        ndim = len(self._mesh._shape)
        if self._degree == ndim:
            return self._cell_restrict(f)
        assert isinstance(f, OneForm), (
            f"degree={self._degree} restriction requires a OneForm, "
            f"got {type(f).__name__}"
        )
        return self._face_restrict(f)

    def _cell_restrict(self, f: SymbolicFunction) -> DiscreteField[sympy.Expr]:
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

    def _face_restrict(self, F: OneForm) -> FaceField[sympy.Expr]:
        mesh = self._mesh
        ndim = len(mesh._shape)

        def face_flux(face: tuple[int, tuple[int, ...]]) -> sympy.Expr:
            axis, idx_low = face
            expr = F.component(axis)
            face_x = (
                mesh._origin[axis]
                + sympy.Integer(idx_low[axis] + 1) * mesh._spacing[axis]
            )
            expr = expr.subs(F.symbols[axis], face_x)
            for j in range(ndim):
                if j != axis:
                    lo = mesh._origin[j] + sympy.Integer(idx_low[j]) * mesh._spacing[j]
                    hi = lo + mesh._spacing[j]
                    expr = sympy.integrate(expr, (F.symbols[j], lo, hi))
            return sympy.simplify(expr)

        return _CallableFaceField(mesh, face_flux)


__all__ = ["CartesianRestrictionOperator"]
