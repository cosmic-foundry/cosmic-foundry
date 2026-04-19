"""LocatedDiscretization ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from cosmic_foundry.theory.discretization import Discretization
from cosmic_foundry.theory.smooth_manifold import SmoothManifold


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
        manifold       — the SmoothManifold this discretization approximates
        node_positions — 1-D coordinate array of DOF positions along *axis*
    """

    @property
    @abstractmethod
    def manifold(self) -> SmoothManifold:
        """The manifold this discretization approximates."""

    @abstractmethod
    def node_positions(self, axis: int) -> Any:
        """1-D coordinate array of DOF positions along *axis*."""


__all__ = [
    "LocatedDiscretization",
]
