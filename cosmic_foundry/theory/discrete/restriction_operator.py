"""RestrictionOperator ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction

F = TypeVar("F")
V = TypeVar("V")


class RestrictionOperator(NumericFunction[F, DiscreteField[V]]):
    """The restriction operator Rₕ: the formal bridge from continuous to discrete.

    A RestrictionOperator maps a continuous function to a DiscreteField.
    The output cochain level (and therefore which field type is produced)
    is fixed by each concrete subclass — the return type annotation of
    __call__ encodes the DEC degree k, making a separate degree property
    redundant.

    The input type F is left generic so that concrete subclasses can narrow
    it (e.g. ZeroForm, OneForm) without an LSP violation.

    The output DiscreteField has .mesh == Rₕ.mesh by construction — the
    discrete values are indexed by the cells (or faces/edges/vertices) of
    Rₕ.mesh and can live on no other mesh.

    Required:
        mesh    — the mesh defining the cell decomposition for restriction
        __call__ — apply the restriction (inherited from NumericFunction)
    """

    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """The mesh defining the cell decomposition for restriction."""


__all__ = ["RestrictionOperator"]
