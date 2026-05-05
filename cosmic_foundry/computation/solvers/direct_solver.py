"""DirectSolver: LinearSolver backed by a Decomposition."""

from __future__ import annotations

from typing import ClassVar

from cosmic_foundry.computation.algorithm_capabilities import (
    LinearOperatorEvidence,
    SmallLinearOperator,
)
from cosmic_foundry.computation.decompositions.decomposition import Decomposition
from cosmic_foundry.computation.solvers.linear_solver import (
    LinearOperator,
    LinearSolver,
)
from cosmic_foundry.computation.solvers.relations import LinearResidualRelation
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

    decomposition_type: ClassVar[type[Decomposition] | None] = None

    def __init__(self, decomposition: Decomposition | None = None) -> None:
        if decomposition is None:
            if self.decomposition_type is None:
                raise TypeError("DirectSolver requires a decomposition")
            decomposition = self.decomposition_type()
        self._decomposition = decomposition

    def _assembled_matrix(
        self, op: SmallLinearOperator, b: Tensor
    ) -> tuple[tuple[float, ...], ...]:
        """Build the N×N matrix by applying op to each basis vector."""
        n = b.shape[0]
        backend = b.backend
        columns: list[list[float]] = []
        for j in range(n):
            e_j = Tensor.zeros(n, backend=backend)
            e_j = e_j.set(j, Tensor(1.0, backend=backend))
            columns.append(backend.flatten(op.apply(e_j)._value))
        # columns[j][i] = A[i, j]; transpose to rows[i][j]
        return tuple(tuple(columns[j][i] for j in range(n)) for i in range(n))

    def _assembled_relation(
        self, op: LinearOperator, b: Tensor
    ) -> LinearResidualRelation:
        matrix = self._assembled_matrix(op, b)
        return LinearResidualRelation(LinearOperatorEvidence(op, b, matrix))

    def _assemble(self, op: LinearOperator, b: Tensor) -> Tensor:
        """Return the dense matrix assembled from a linear operator."""
        return Tensor(self._assembled_matrix(op, b), backend=b.backend)

    def solve_relation(self, relation: LinearResidualRelation) -> Tensor:
        """Solve an assembled linear residual relation."""
        evidence = relation.linear_operator_evidence
        matrix = evidence.matrix or self._assembled_matrix(
            evidence.operator, evidence.rhs
        )
        a: Tensor = Tensor(matrix, backend=evidence.rhs.backend)
        return self._decomposition.decompose(a).solve(evidence.rhs)

    def solve(self, op: LinearOperator, b: Tensor) -> Tensor:
        return self.solve_relation(self._assembled_relation(op, b))


__all__ = ["DirectSolver"]
