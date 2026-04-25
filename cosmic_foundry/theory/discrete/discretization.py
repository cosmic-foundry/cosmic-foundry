"""Discretization ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cosmic_foundry.theory.discrete.mesh import Mesh


class Discretization(ABC):
    """Encapsulates a discrete scheme on a mesh.

    A Discretization holds the scheme choice — reconstruction, numerical
    flux, quadrature — for a particular mesh and approximation order.
    Calling it produces the DiscreteOperator Lₕ that makes the commutation
    diagram

        Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))

    hold, where p is the approximation order.

    Required:
        mesh     — the mesh on which the scheme is defined
        __call__ — produce the DiscreteOperator (signature defined by subclass)
    """

    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """The mesh on which the scheme is defined."""


__all__ = ["Discretization"]
