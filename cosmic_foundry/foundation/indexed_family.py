"""IndexedFamily ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from cosmic_foundry.foundation.set import Set


class IndexedFamily(Set):
    """A finite family of objects indexed by a contiguous integer range {0, …, n-1}.

    An IndexedFamily is a Set whose elements are retrievable by integer index.
    It is the abstract type for any finite ordered collection: a tuple of
    patches, a distributed field of arrays, a list of mesh blocks.

    Two operations are required:
        __getitem__(i) — retrieve the element at index i
        __len__()      — total number of elements

    IndexedFamily is distinct from IndexedSet: IndexedSet describes an
    integer *index space* (the domain of a discretization), while
    IndexedFamily describes a *collection of objects* that happen to be
    accessed by integer index.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return the element at *index*."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of elements."""


__all__ = [
    "IndexedFamily",
]
