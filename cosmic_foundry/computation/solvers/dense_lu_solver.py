"""DenseLUSolver: DirectSolver backed by LUFactorization."""

from __future__ import annotations

from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver


class DenseLUSolver(DirectSolver):
    """Direct solver for A u = b using LU factorization with partial pivoting.

    Convenience wrapper around DirectSolver(LUFactorization()).
    See LUFactorization for algorithm details.
    """

    decomposition_type = LUFactorization


__all__ = ["DenseLUSolver"]
