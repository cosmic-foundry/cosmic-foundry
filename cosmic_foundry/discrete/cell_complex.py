"""CellComplex ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.foundation.function import Function
from cosmic_foundry.foundation.indexed_family import IndexedFamily
from cosmic_foundry.foundation.set import Set


class CellComplex(IndexedFamily):
    """A chain complex (C_*, ∂): a graded family of cell sets with boundary operators.

    A CellComplex is an IndexedFamily whose elements are the sets of
    k-cells for k = 0, …, n.  It adds boundary operators ∂_k: C_k → C_{k-1}
    satisfying the algebraic identity ∂² = 0 (∂_{k-1} ∘ ∂_k = 0), the
    identity underlying the divergence theorem.

    Example — 2D Cartesian N×M grid:
        C_0: (N+1)(M+1) vertices
        C_1: N(M+1) horizontal + (N+1)M vertical edges
        C_2: N×M cells
        ∂₁: signed vertex-incidence; ∂₂: signed edge-incidence

    Required:
        __getitem__ — return the Set of k-cells (narrows IndexedFamily to Set)
        __len__     — number of cell dimensions (n+1 for an n-dimensional complex)
        boundary    — return the boundary operator ∂_k: C_k → C_{k-1}
    """

    @abstractmethod
    def __getitem__(self, k: int) -> Set:
        """Return the Set of k-cells."""

    @abstractmethod
    def boundary(self, k: int) -> Function:
        """Return the boundary operator ∂_k: C_k → C_{k-1}."""


__all__ = ["CellComplex"]
