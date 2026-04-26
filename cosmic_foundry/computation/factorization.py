"""Factorization ABC and FactoredMatrix ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cosmic_foundry.computation.tensor import Tensor


class FactoredMatrix(ABC):
    """A factored matrix that can solve A u = b for arbitrary b."""

    @abstractmethod
    def solve(self, b: Tensor) -> Tensor:
        """Solve the factored system for rhs b; return u."""


class Factorization(ABC):
    """A matrix factorization algorithm.

    A Factorization accepts a square matrix A and returns a FactoredMatrix
    that can solve A u = b for any right-hand side b.  The factorization step
    (O(N³) for LU) is paid once; each subsequent solve costs O(N²).

    Required:
        factorize — factor A and return a FactoredMatrix
    """

    @abstractmethod
    def factorize(self, a: Tensor) -> FactoredMatrix:
        """Factor A; return a FactoredMatrix that can solve A u = b."""


__all__ = ["Factorization", "FactoredMatrix"]
