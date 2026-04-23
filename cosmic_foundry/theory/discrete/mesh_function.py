"""MeshFunction ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction

V = TypeVar("V")


class MeshFunction(NumericFunction[Mesh, V], Generic[V]):
    """A value assignment to mesh elements: cells, faces, or vertices.

    A MeshFunction is a NumericFunction whose domain is a Mesh.  It earns
    its class via the typed accessor .mesh: Mesh, by analogy with
    Field.manifold — the mesh constrains which elements carry values.

    Required:
        mesh    — the mesh on which this function's values are defined
        __call__ — evaluate the function (inherited from NumericFunction)
    """

    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """The mesh on which this function's values are defined."""


__all__ = ["MeshFunction"]
