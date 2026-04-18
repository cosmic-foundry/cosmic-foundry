"""Domain ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Domain(ABC):
    """Abstract base for all domain types: the set D over which fields are defined.

    A domain is the input space of a field f: D → ℝ. Domains differ in their
    representation (continuous vs. discrete) and their nature (physical space,
    thermodynamic state space, etc.). Every Field has a Domain; a Domain is
    not itself a Field.

    The one universal property of a domain is its dimensionality.
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of dimensions of this domain."""


__all__ = [
    "Domain",
]
