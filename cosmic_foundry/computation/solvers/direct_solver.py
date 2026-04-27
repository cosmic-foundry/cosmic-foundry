"""DirectSolver: LinearSolver backed by a Decomposition."""

from __future__ import annotations

from cosmic_foundry.computation.decompositions.decomposition import Decomposition
from cosmic_foundry.computation.solvers.linear_solver import LinearSolver
from cosmic_foundry.computation.tensor import Tensor


class DirectSolver(LinearSolver):
    """LinearSolver that delegates to a Decomposition.

    solve(a, b) decomposes A once and then solves by applying the stored
    DecomposedTensor.  Cost is dominated by the decomposition step: O(N³)
    for LU and SVD.  The decomposed form is not cached; each solve call
    re-decomposes A.  For problems where A is constant across time steps,
    the caller should reuse the DecomposedTensor directly.

    Parameters
    ----------
    decomposition:
        The Decomposition algorithm to use.
    """

    def __init__(self, decomposition: Decomposition) -> None:
        self._decomposition = decomposition

    def solve(self, a: Tensor, b: Tensor) -> Tensor:
        return self._decomposition.decompose(a).solve(b)


__all__ = ["DirectSolver"]
