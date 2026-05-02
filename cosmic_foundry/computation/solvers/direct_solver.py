"""DirectSolver: LinearSolver backed by a Decomposition."""

from __future__ import annotations

from cosmic_foundry.computation.decompositions.decomposition import Decomposition
from cosmic_foundry.computation.solvers._capability_claims import (
    LinearSolverCapability,
    contract,
)
from cosmic_foundry.computation.solvers.linear_solver import (
    LinearOperator,
    LinearSolver,
)
from cosmic_foundry.computation.tensor import Tensor


class DirectSolver(LinearSolver):
    """LinearSolver that assembles the matrix then delegates to a Decomposition.

    solve(op, b) builds the N×N stiffness matrix by calling op.apply on each
    of the N standard basis vectors, then decomposes and solves.  Cost is
    dominated by the decomposition step: O(N³) for LU and SVD.  The assembled
    matrix is not cached; each solve call reassembles and re-decomposes.  For
    problems where A is constant across time steps, obtain the factorization
    directly via the Decomposition class and reuse it.

    Parameters
    ----------
    decomposition:
        The Decomposition algorithm to use.
    """

    def __init__(self, decomposition: Decomposition) -> None:
        self._decomposition = decomposition

    def _assemble(self, op: LinearOperator, b: Tensor) -> Tensor:
        """Build the N×N matrix by applying op to each basis vector."""
        n = b.shape[0]
        backend = b.backend
        columns: list[list[float]] = []
        for j in range(n):
            e_j = Tensor.zeros(n, backend=backend)
            e_j = e_j.set(j, Tensor(1.0, backend=backend))
            columns.append(backend.flatten(op.apply(e_j)._value))
        # columns[j][i] = A[i, j]; transpose to rows[i][j]
        rows = [[columns[j][i] for j in range(n)] for i in range(n)]
        return Tensor(rows, backend=backend)

    def solve(self, op: LinearOperator, b: Tensor) -> Tensor:
        a = self._assemble(op, b)
        return self._decomposition.decompose(a).solve(b)


def declare_linear_solver_capabilities() -> tuple[LinearSolverCapability, ...]:
    """Return capability declarations owned by this solver implementation."""
    return (
        LinearSolverCapability(
            "generic_direct",
            "DirectSolver",
            "direct_solver",
            contract(
                requires=("linear_operator", "decomposition"),
                provides=("solve", "direct", "assembled_matrix"),
            ),
        ),
    )


__all__ = ["DirectSolver", "declare_linear_solver_capabilities"]
