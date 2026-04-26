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
    The input type F is left generic so that concrete subclasses can
    narrow it (e.g. SymbolicFunction) without an LSP violation.

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

    @property
    @abstractmethod
    def degree(self) -> int:
        """DEC degree k of the discrete values produced by this restriction.

        degree == ndim restricts a ZeroForm to cell volume integrals (n-chains).
        degree == ndim - 1 restricts a OneForm's normal component to face
        integrals ((n-1)-chains).
        """


__all__ = ["RestrictionOperator"]
