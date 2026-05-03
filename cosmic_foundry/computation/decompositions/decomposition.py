"""DecomposedTensor ABC and Decomposition ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from cosmic_foundry.computation.algorithm_capabilities import StructuredPredicate
from cosmic_foundry.computation.tensor import Tensor


class DecomposedTensor(ABC):
    """A decomposed tensor that can solve A u = b for arbitrary b.

    DecomposedTensor is the general result of any matrix decomposition.
    The solve semantics depend on the concrete decomposition:

    - Factorizations (LU, Cholesky, QR): exact solution; minimum-norm for
      singular systems via zero-pinning or rank truncation.
    - SVD: Moore-Penrose pseudoinverse; minimum-norm least-squares solution
      with explicit rank threshold.

    Required:
        solve — apply the stored decomposition to rhs b and return u
    """

    @abstractmethod
    def solve(self, b: Tensor) -> Tensor:
        """Solve the decomposed system for rhs b; return u."""


class Decomposition(ABC):
    """A matrix decomposition algorithm.

    A Decomposition accepts a matrix A and returns a DecomposedTensor that
    can solve A u = b for any right-hand side b.  The decomposition step
    (O(N³) for most dense algorithms) is paid once; each subsequent solve
    costs O(N²) or cheaper.

    The two concrete families are:
    - Factorization: product form A = F₁ F₂ … Fₖ (LU, Cholesky, QR).
      Exact solution for full-rank square systems.
    - SVDDecomposition: sum form A = U Σ Vᵀ.
      Minimum-norm pseudoinverse; handles rank-deficient systems.

    Required:
        decompose — decompose A and return a DecomposedTensor
    """

    factorization_feasibility_certificate: ClassVar[
        tuple[StructuredPredicate, ...]
    ] = ()

    @abstractmethod
    def decompose(self, a: Tensor) -> DecomposedTensor:
        """Decompose A; return a DecomposedTensor that can solve A u = b."""


__all__ = ["Decomposition", "DecomposedTensor"]
