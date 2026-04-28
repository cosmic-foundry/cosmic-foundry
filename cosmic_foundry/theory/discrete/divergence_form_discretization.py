"""DivergenceFormDiscretization: discretizes a linear operator factored as L = ∇·f."""

from __future__ import annotations

from typing import Any, cast

import sympy

from cosmic_foundry.geometry.cartesian_exterior_derivative import (
    CartesianExteriorDerivative,
)
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_operator import (
    DifferentialOperator,
    DivergenceComposition,
)
from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DiscreteBoundaryCondition,
)
from cosmic_foundry.theory.discrete.discrete_field import (
    DiscreteField,
    _CallableDiscreteField,
)
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


class DivergenceFormDiscretization(Discretization[sympy.Expr]):
    """Discretization of a linear operator factored as L = ∇·f.

    Given a NumericalFlux that discretizes f: state → face values, builds the
    discrete operator Lₕ acting on cell-average DOFs:

        Lₕ U = (1/vol) · d_{n−1}(F̂(bc.extend(U)))

    where F̂ is the NumericalFlux, d_{n−1} is the discrete exterior derivative
    (CartesianExteriorDerivative at degree=2), and /vol normalizes from
    cell-total to cell-average DOFs.

    The "flux" is a formal intermediate quantity at faces — a local function
    of cell averages, defined for the convenience of factoring L as ∇·f.  It
    does not represent transfer of conserved quantity between cells; the
    equations we solve (Poisson, steady advection, steady advection-diffusion)
    are elliptic algebraic constraints, not time evolutions.

    The unique role of this class is to package (NumericalFlux, BC) into a
    Discretization with the appropriate continuous_operator declaration.  All
    operational steps delegate to existing machinery (BC.extend, NumericalFlux,
    CartesianExteriorDerivative, mesh.cell_volume).

    Parameters
    ----------
    numerical_flux:
        The NumericalFlux discretizing the flux operator f.  Determines the
        convergence order and the continuous operator being approximated:
        continuous_operator = ∇·(numerical_flux.continuous_operator).
    boundary_condition:
        DiscreteBoundaryCondition; defaults to ZeroGhostCells() (absorbing).
    """

    def __init__(
        self,
        numerical_flux: NumericalFlux[Any],
        boundary_condition: DiscreteBoundaryCondition | None = None,
    ) -> None:
        super().__init__(boundary_condition)
        self._numerical_flux = numerical_flux

    @property
    def order(self) -> int:
        return self._numerical_flux.order

    @property
    def continuous_operator(self) -> DifferentialOperator:
        return DivergenceComposition(self._numerical_flux.continuous_operator)

    def __call__(self, U: DiscreteField[sympy.Expr]) -> DiscreteField[sympy.Expr]:
        """Apply Lₕ U: discretization of ∇·f at cell-average DOFs."""
        mesh = cast(CartesianMesh, U.mesh)
        U = self._boundary_condition.extend(U, mesh)
        face_fluxes = self._numerical_flux(U)
        vol = mesh.cell_volume
        div = CartesianExteriorDerivative(mesh, degree=2)(face_fluxes)
        return _CallableDiscreteField(mesh, lambda idx: div(idx) / vol)


__all__ = ["DivergenceFormDiscretization"]
