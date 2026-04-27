"""DirectSolver: LinearSolver backed by a Factorization."""

from __future__ import annotations

from cosmic_foundry.computation.factorization import Factorization
from cosmic_foundry.computation.linear_solver import LinearSolver
from cosmic_foundry.computation.tensor import Tensor


class DirectSolver(LinearSolver):
    """LinearSolver that delegates to a Factorization.

    solve(a, b) factors A once and then solves by substitution.
    Cost is dominated by the factorization step: O(N³) for LU,
    O(N³/3) for Cholesky.  The factored form is not cached; each
    solve call re-factors A.  For problems where A is constant across
    time steps, the caller should reuse the FactoredMatrix directly.

    Parameters
    ----------
    factorization:
        The Factorization algorithm to use.
    """

    cost_exponent = 3  # O(N^3) factorization dominates the O(N^2) substitution

    def __init__(self, factorization: Factorization) -> None:
        self._factorization = factorization

    def solve(self, a: Tensor, b: Tensor) -> Tensor:
        return self._factorization.factorize(a).solve(b)


__all__ = ["DirectSolver"]
