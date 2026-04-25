"""PeriodicBC: periodic boundary condition φ(x + L) = φ(x) on ∂Ω."""

from __future__ import annotations

from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.continuous.non_local_boundary_condition import (
    NonLocalBoundaryCondition,
)


class PeriodicBC(NonLocalBoundaryCondition):
    """Periodic boundary condition: φ(x + L) = φ(x) on ∂Ω.

    Identifies each low boundary face with the opposite high boundary face along
    every axis.  The ghost-cell rule for FVM discretization wraps indices modulo
    the mesh shape: a cell at index −k aliases cell N−k, and cell at index N+k
    aliases cell k.

    For pure-advection operators the assembled stiffness matrix has a one-
    dimensional null space spanned by the constant field (zero eigenvalue of the
    circulant matrix).  The equation ∇·(vφ) = f is solvable under PeriodicBC
    only when ∫f = 0 (compatibility condition); DenseLUSolver satisfies this
    by choosing the minimum-norm (zero-mean) solution when it encounters a
    near-zero pivot.

    Parameters
    ----------
    manifold:
        The manifold on which this BC is posed.
    """

    def __init__(self, manifold: Manifold) -> None:
        self._manifold = manifold

    @property
    def manifold(self) -> Manifold:
        return self._manifold

    @property
    def support(self) -> Manifold:
        return self._manifold


__all__ = ["PeriodicBC"]
