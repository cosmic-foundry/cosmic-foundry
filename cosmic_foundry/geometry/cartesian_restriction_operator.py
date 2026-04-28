"""Cartesian restriction operators: analytic Rₕᵏ on CartesianMesh."""

from __future__ import annotations

from itertools import product

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ZeroForm,
)
from cosmic_foundry.theory.discrete.edge_field import EdgeField, _CallableEdgeField
from cosmic_foundry.theory.discrete.face_field import FaceField, _CallableFaceField
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.point_field import PointField, _CallablePointField
from cosmic_foundry.theory.discrete.restriction_operator import RestrictionOperator
from cosmic_foundry.theory.discrete.volume_field import VolumeField


class _CartesianVolumeIntegral(VolumeField[sympy.Expr]):
    """Cell volume integrals on a CartesianMesh."""

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


class CartesianVolumeRestriction(RestrictionOperator[DifferentialForm, sympy.Expr]):
    """Rₕⁿ: n-Form → VolumeField via exact SymPy cell-volume integration.

    (Rₕⁿ f)ᵢ = ∫_Ωᵢ f dV — total cell integral, returned as a VolumeField.
    Cell averages are VolumeField values divided by cell_volume.

    Input must be an n-Form whose degree equals the mesh dimension
    (OneForm in 1-D, TwoForm in 2-D, ThreeForm in 3-D).  Valid in
    Cartesian coordinates where the Jacobian is 1; non-Cartesian metrics
    must fold √|g| into f before calling.

    This is the FV restriction: the choice of cell-average DOFs.
    """

    def __init__(self, mesh: CartesianMesh) -> None:
        self._mesh = mesh

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, f: DifferentialForm) -> VolumeField[sympy.Expr]:
        ndim = len(self._mesh._shape)
        assert isinstance(f, DifferentialForm) and f.degree == ndim, (
            f"CartesianVolumeRestriction requires an n-Form of degree {ndim}, "
            f"got {type(f).__name__}"
        )
        mesh = self._mesh
        values: dict[tuple[int, ...], sympy.Expr] = {}
        for idx in product(*[range(s) for s in mesh._shape]):
            expr = f.expr
            for i, sym in enumerate(f.symbols):
                lo = mesh._origin[i] + sympy.Integer(idx[i]) * mesh._spacing[i]
                hi = lo + mesh._spacing[i]
                expr = sympy.integrate(expr, (sym, lo, hi))
            values[idx] = sympy.simplify(expr)
        return _CartesianVolumeIntegral(mesh, values)


class CartesianFaceRestriction(RestrictionOperator[OneForm, sympy.Expr]):
    """Rₕⁿ⁻¹: OneForm → FaceField via exact SymPy face-normal integration.

    (Rₕⁿ⁻¹ F)_{a,i} = ∫_{transverse} F.component(a)|_{x_a=face} dx_⊥ → FaceField.

    OneForm is used as a Cartesian (n-1)-Form proxy: F.component(a) stands
    for *(F)_a dA_⊥.  Commutation Dₙ₋₁ ∘ Rₕⁿ⁻¹ = Rₕⁿ ∘ dₙ₋₁ holds exactly
    (Stokes theorem).
    """

    def __init__(self, mesh: CartesianMesh) -> None:
        self._mesh = mesh

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, F: OneForm) -> FaceField[sympy.Expr]:
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


class CartesianEdgeRestriction(RestrictionOperator[OneForm, sympy.Expr]):
    """Rₕ¹: OneForm → EdgeField via exact SymPy edge-tangent integration.

    (Rₕ¹ F)_{a,c} = ∫_{x_c}^{x_{c+1}} F.component(a)|_{x_⊥=x_{c_⊥}} dx_a → EdgeField.

    Integrates over the full cell width [x_c, x_{c+1}]; transverse coordinates
    fixed at the low-face position.  Only meaningful for ndim > 1; for ndim == 1
    this coincides with CartesianVolumeRestriction.
    Commutation D₀ ∘ Rₕ⁰ = Rₕ¹ ∘ d₀ holds exactly (FTC).
    """

    def __init__(self, mesh: CartesianMesh) -> None:
        self._mesh = mesh

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, F: OneForm) -> EdgeField[sympy.Expr]:
        mesh = self._mesh
        ndim = len(mesh._shape)

        def edge_integral(edge: tuple[int, tuple[int, ...]]) -> sympy.Expr:
            axis, c_idx = edge
            expr = F.component(axis)
            lo = mesh._origin[axis] + sympy.Integer(c_idx[axis]) * mesh._spacing[axis]
            hi = lo + mesh._spacing[axis]
            for j in range(ndim):
                if j != axis:
                    x_cj = mesh._origin[j] + sympy.Integer(c_idx[j]) * mesh._spacing[j]
                    expr = expr.subs(F.symbols[j], x_cj)
            expr = sympy.integrate(expr, (F.symbols[axis], lo, hi))
            return sympy.simplify(expr)

        return _CallableEdgeField(mesh, edge_integral)


class CartesianPointRestriction(RestrictionOperator[ZeroForm, sympy.Expr]):
    """Rₕ⁰: ZeroForm → PointField via cell-center evaluation.

    (Rₕ⁰ f)(c) = f(origin + (c + ½)·spacing) → PointField.

    This is the FD restriction: the choice of point-value DOFs at cell centers.
    Commutation D₀ ∘ Rₕ⁰ = Rₕ¹ ∘ d₀ holds exactly (FTC).
    """

    def __init__(self, mesh: CartesianMesh) -> None:
        self._mesh = mesh

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, f: ZeroForm) -> PointField[sympy.Expr]:
        mesh = self._mesh
        ndim = len(mesh._shape)

        def point_eval(c_idx: tuple[int, ...]) -> sympy.Expr:
            expr = f.expr
            for j in range(ndim):
                x_cj = (
                    mesh._origin[j]
                    + (sympy.Integer(c_idx[j]) + sympy.Rational(1, 2))
                    * mesh._spacing[j]
                )
                expr = expr.subs(f.symbols[j], x_cj)
            return sympy.simplify(expr)

        return _CallablePointField(mesh, point_eval)


__all__ = [
    "CartesianEdgeRestriction",
    "CartesianFaceRestriction",
    "CartesianPointRestriction",
    "CartesianVolumeRestriction",
]
