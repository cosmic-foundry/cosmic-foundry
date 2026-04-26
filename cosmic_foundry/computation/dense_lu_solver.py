"""DenseLUSolver: DirectSolver backed by LUFactorization."""

from __future__ import annotations

from cosmic_foundry.computation.direct_solver import DirectSolver
from cosmic_foundry.computation.lu_factorization import LUFactorization


class DenseLUSolver(DirectSolver):
    """Direct solver for A u = b using LU factorization with partial pivoting.

    Convenience wrapper around DirectSolver(LUFactorization()).
    See LUFactorization for algorithm details.
    """

    def __init__(self) -> None:
        super().__init__(LUFactorization())


__all__ = ["DenseLUSolver"]
