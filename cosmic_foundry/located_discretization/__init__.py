"""LocatedDiscretization ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from cosmic_foundry.discretization import Discretization


class LocatedDiscretization(Discretization):
    """A Discretization whose DOFs are located at specific points in M.

    Each index i ∈ I is mapped to a physical point φ(i) ∈ M by a
    coordinate function φ: I → M.  This is the natural abstraction for
    any method where field values are stored at — or reconstructed at —
    specific positions: finite-difference cell centers, finite-element
    nodes, finite-volume cell centroids, and particle positions all live
    here.

    Concrete subclasses differ in how φ is represented:
    - Patch: φ(i) = origin + i·h  (uniform Cartesian formula)
    - ParticleSet: φ(i) = positions[i]  (explicit array)

    Required:
        node_positions(axis) — 1-D coordinate array of DOF positions
                               along *axis* in physical space
    """

    @abstractmethod
    def node_positions(self, axis: int) -> Any:
        """1-D coordinate array of DOF positions along *axis*."""


__all__ = [
    "LocatedDiscretization",
]
