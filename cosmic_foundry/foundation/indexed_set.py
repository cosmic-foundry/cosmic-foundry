"""IndexedSet ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.foundation.set import Set


class IndexedSet(Set):
    """A Set equipped with a bijection to a finite subset of ℤⁿ.

    An IndexedSet adds exactly one piece of structure to a Set: its elements
    can be put in bijection with a contiguous rectangular region of the integer
    lattice ℤⁿ.  This bijection gives each element a unique multi-index
    (i₁, …, iₙ) ∈ ℤⁿ and endows the set with a notion of shape and
    dimensionality, but not yet of geometry or coordinates in physical space.

    Subclasses that add physical coordinates (e.g. Discretization) inherit
    from IndexedSet and layer on additional structure.

    Required:
        ndim      — number of index dimensions n
        shape     — sizes (s₁, …, sₙ) of the index range along each axis,
                    so that the total number of elements is ∏ sᵢ
        intersect — set-theoretic intersection: S ∩ T → IndexedSet | None
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of index dimensions."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Size along each index axis."""

    @abstractmethod
    def intersect(self, other: IndexedSet) -> IndexedSet | None:
        """Return the intersection S ∩ T, or None if the sets are disjoint."""


__all__ = [
    "IndexedSet",
]
