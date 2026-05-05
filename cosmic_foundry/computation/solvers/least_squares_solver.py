"""Least-squares solver interface and dense SVD implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cosmic_foundry.computation.decompositions.svd_factorization import SVDFactorization
from cosmic_foundry.computation.solvers.relations import LeastSquaresRelation
from cosmic_foundry.computation.tensor import Tensor


class LeastSquaresSolver(ABC):
    """Abstract interface for solving min_x ||A x - b||_2."""

    @abstractmethod
    def solve(self, relation: LeastSquaresRelation) -> Tensor:
        """Return the least-squares solution for a residual relation."""


class DenseSVDLeastSquaresSolver(LeastSquaresSolver):
    """Dense least-squares solver backed by the Moore-Penrose pseudoinverse."""

    def __init__(self, rcond: float = 1e-10) -> None:
        self._factorization = SVDFactorization(rcond)

    def solve(self, relation: LeastSquaresRelation) -> Tensor:
        """Return x minimizing ||A x - b||_2."""
        evidence = relation.linear_operator_evidence
        a: Tensor = Tensor(evidence.matrix, backend=evidence.rhs.backend)
        return self._factorization.factorize(a).solve(evidence.rhs)


__all__ = ["DenseSVDLeastSquaresSolver", "LeastSquaresSolver"]
