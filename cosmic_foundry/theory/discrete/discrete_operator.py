"""DiscreteOperator ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction

_V = TypeVar("_V")


class DiscreteOperator(NumericFunction[MeshFunction[_V], MeshFunction[_V]]):
    """The discrete analog of DifferentialOperator: Lₕ: MeshFunction → MeshFunction.

    A DiscreteOperator is the output of a Discretization — the Lₕ that makes
    the commutation diagram Lₕ ∘ Rₕ ≈ Rₕ ∘ L hold to the chosen approximation
    order.  It earns its class via the typed accessor .mesh: Mesh, which
    constrains both input and output to the same mesh, by analogy with
    DifferentialOperator.manifold in the continuous layer.

    A DiscreteOperator is not constructed directly from stencil coefficients;
    it is produced by a Discretization.

    Required:
        mesh     — the mesh shared by input and output MeshFunctions
        __call__ — apply the operator (inherited from NumericFunction)
    """

    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """The mesh shared by input and output MeshFunctions."""


__all__ = ["DiscreteOperator"]
