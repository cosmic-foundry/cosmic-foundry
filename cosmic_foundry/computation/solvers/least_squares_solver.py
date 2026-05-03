"""Least-squares solver interface and dense SVD implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cosmic_foundry.computation.decompositions.svd_factorization import SVDFactorization
from cosmic_foundry.computation.tensor import Tensor


class LeastSquaresSolver(ABC):
    """Abstract interface for solving min_x ||A x - b||_2."""

    @abstractmethod
    def solve(self, a: Tensor, b: Tensor) -> Tensor:
        """Return the least-squares solution for a matrix A and target b."""


class DenseSVDLeastSquaresSolver(LeastSquaresSolver):
    """Dense least-squares solver backed by the Moore-Penrose pseudoinverse."""

    def __init__(self, rcond: float = 1e-10) -> None:
        self._factorization = SVDFactorization(rcond)

    def solve(self, a: Tensor, b: Tensor) -> Tensor:
        """Return x minimizing ||A x - b||_2."""
        if len(a.shape) != 2 or len(b.shape) != 1 or a.shape[0] != b.shape[0]:
            raise ValueError("least-squares solve requires A shape (m,n), b shape (m,)")
        return self._factorization.factorize(a).solve(b)


__all__ = ["DenseSVDLeastSquaresSolver", "LeastSquaresSolver"]
