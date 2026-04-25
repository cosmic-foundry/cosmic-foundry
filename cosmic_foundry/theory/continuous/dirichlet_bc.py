"""DirichletBC: homogeneous Dirichlet boundary condition φ = 0 on ∂Ω."""

from __future__ import annotations

import sympy

from cosmic_foundry.theory.continuous.differential_form import ZeroForm
from cosmic_foundry.theory.continuous.local_boundary_condition import (
    LocalBoundaryCondition,
)
from cosmic_foundry.theory.continuous.manifold import Manifold


class DirichletBC(LocalBoundaryCondition):
    """Homogeneous Dirichlet boundary condition: φ = 0 on ∂Ω.

    Fixes α = 1, β = 0 in the Robin family α·φ + β·∂φ/∂n = g, with g ≡ 0.
    The ghost-cell rule for FVM discretization follows by odd reflection:
    the cell value at mirror index −1−k reflects interior cell k with a sign
    flip, so the interpolated face value is zero at the boundary.

    Parameters
    ----------
    manifold:
        The manifold on which this BC is posed; used to construct the
        zero-valued constraint field.
    """

    def __init__(self, manifold: Manifold) -> None:
        self._zero: ZeroForm = ZeroForm(manifold, sympy.Integer(0), ())

    @property
    def alpha(self) -> float:
        return 1.0

    @property
    def beta(self) -> float:
        return 0.0

    @property
    def constraint(self) -> ZeroForm:
        """The prescribed boundary data: g ≡ 0."""
        return self._zero


__all__ = ["DirichletBC"]
