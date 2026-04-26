"""DiscreteField ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction

V = TypeVar("V")


class DiscreteField(NumericFunction[Mesh, V]):
    """A value assignment to mesh elements: cells, faces, or vertices.

    A DiscreteField is a NumericFunction whose domain is a Mesh — the discrete
    analog of Field, which maps a manifold to a value.  It earns its class via
    the typed accessor .mesh: Mesh, by analogy with Field.manifold — the mesh
    constrains which elements carry values.

    Required:
        mesh     — the mesh on which this field's values are defined
        __call__ — evaluate the field (inherited from NumericFunction)
    """

    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """The mesh on which this field's values are defined."""


__all__ = ["DiscreteField"]
