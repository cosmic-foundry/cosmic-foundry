"""Factorization ABC — product-form matrix decomposition."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.computation.decompositions.decomposition import (
    DecomposedTensor,
    Decomposition,
)
from cosmic_foundry.computation.tensor import Tensor


class Factorization(Decomposition):
    """Product-form matrix decomposition: A = F₁ F₂ … Fₖ.

    A Factorization expresses A as a product of simpler matrices — L and U
    for LU, L for Cholesky, Q and R for QR.  The factored form is stored in
    a DecomposedTensor that solves A u = b by substitution.

    Factorization is a strict subtype of Decomposition.  Not all
    decompositions are factorizations: the SVD sum form A = U Σ Vᵀ is a
    Decomposition but not a Factorization because it is not expressed as a
    matrix product without inverses.

    The factorize step (O(N³) for LU) is paid once; each subsequent solve
    costs O(N²).

    Required:
        factorize — factor A and return a DecomposedTensor
    """

    @abstractmethod
    def factorize(self, a: Tensor) -> DecomposedTensor:
        """Factor A; return a DecomposedTensor that can solve A u = b."""

    def decompose(self, a: Tensor) -> DecomposedTensor:
        """Decompose A by factorization; delegates to factorize."""
        return self.factorize(a)


__all__ = ["Factorization"]
