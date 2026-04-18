"""Domain — the abstract type of all sets that can serve as the domain of a field."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Domain(ABC):
    """The set D on which a field f: D → ℝ is defined.

    A domain is not a region, a grid, or a spatial extent — it is the abstract
    type of all sets that can serve as the input space of a field.  Domains
    differ in their structure (continuous, discrete, graph-valued, …) and their
    nature (physical space, thermodynamic state space, index space, …).

    Concrete subclasses supply the structure of D.  This ABC makes no
    assumptions about geometry, boundedness, or dimensionality.
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of dimensions of this domain."""


__all__ = [
    "Domain",
]
