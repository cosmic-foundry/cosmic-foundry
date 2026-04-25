"""AdvectionDiffusionFlux: NumericalFlux for F(φ) = φ − κ∇φ."""

from __future__ import annotations

from typing import ClassVar, cast

import sympy

from cosmic_foundry.geometry.advective_flux import AdvectiveFlux
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from cosmic_foundry.theory.continuous.advection_diffusion_operator import (
    AdvectionDiffusionOperator,
)
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


class AdvectionDiffusionFlux(NumericalFlux[sympy.Expr]):
    """Numerical flux for the advection-diffusion equation F(φ) = φ − κ∇φ.

    AdvectionDiffusionFlux approximates the face-averaged normal flux
    (φ(face) − κ·∂φ/∂xₐ(face))·|Aₐ| at each cell interface.  The advective
    part uses a symmetric centered reconstruction (AdvectiveFlux) and the
    diffusive part uses an antisymmetric finite-difference stencil
    (DiffusiveFlux), both at the same convergence order.

    The assembled stiffness matrix is A = A_adv + κ·A_diff, where A_adv is
    skew-symmetric and A_diff is symmetric positive definite (DirichletBC).
    The matrix is non-singular for any κ > 0 under DirichletBC, so both
    DenseJacobiSolver and DenseLUSolver are compatible.  Jacobi convergence
    is guaranteed when κ is large enough relative to the cell spacing that the
    SPD part dominates (Péclet number Pe = h/(2κ) < 1).

    Parameters
    ----------
    order:
        Scheme order, satisfying order >= 2 and (order − 2) % 2 == 0.
        Both the advective and diffusive stencils are built at this order,
        so the combined flux achieves O(hᵖ) accuracy.
    manifold:
        The manifold on which the operator acts.
    kappa:
        Diffusion coefficient (default: 1).  Must be a positive sympy.Expr or
        a value convertible to one; the numerical stencil scales the diffusive
        face flux by this factor.
    """

    min_order: ClassVar[int] = 2
    order_step: ClassVar[int] = 2

    def __init__(
        self,
        order: int,
        manifold: Manifold,
        kappa: sympy.Expr | None = None,
    ) -> None:
        if order < self.min_order or (order - self.min_order) % self.order_step != 0:
            raise ValueError(
                f"AdvectionDiffusionFlux order must be >= {self.min_order} and satisfy "
                f"(order - {self.min_order}) % {self.order_step} == 0; got {order}"
            )
        self._order = order
        self._kappa: sympy.Expr = sympy.Integer(1) if kappa is None else kappa
        self._adv_flux = AdvectiveFlux(order, manifold)
        self._diff_flux = DiffusiveFlux(order, manifold)
        self._continuous_operator = AdvectionDiffusionOperator(manifold, self._kappa)

    @property
    def order(self) -> int:
        return self._order

    @property
    def continuous_operator(self) -> DifferentialOperator:
        return self._continuous_operator

    def __call__(
        self,
        U: MeshFunction[sympy.Expr],
    ) -> LazyMeshFunction[sympy.Expr]:
        """Return a face-flux MeshFunction over all faces.

        The returned MeshFunction is callable as result((axis, idx_low))
        where idx_low is the low-side cell index tuple.  The face flux is the
        sum of the advective face reconstruction and κ times the diffusive
        face derivative.
        """
        mesh = cast(CartesianMesh, U.mesh)
        kappa = self._kappa
        adv = self._adv_flux(U)
        diff = self._diff_flux(U)

        def compute(face: tuple[int, tuple[int, ...]]) -> sympy.Expr:
            return adv(face) + kappa * diff(face)

        return LazyMeshFunction(mesh, compute)


__all__ = ["AdvectionDiffusionFlux"]
