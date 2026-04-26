"""RestrictionOperator ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.foundation.function import Function
from cosmic_foundry.theory.foundation.numeric_function import NumericFunction

M = TypeVar("M")
V = TypeVar("V")


class RestrictionOperator(NumericFunction[Function[M, V], DiscreteField[V]]):
    """The restriction operator Rₕ: the formal bridge from continuous to discrete.

    A RestrictionOperator maps a continuous Function to a DiscreteField via
    cell-averaged integration:

        (Rₕ f)ᵢ = |Ωᵢ|⁻¹ ∫_Ωᵢ f dV

    The output DiscreteField has .mesh == Rₕ.mesh by construction — the
    cell averages are indexed by the cells of Rₕ.mesh and can live on
    no other mesh.

    When f is a Field (SymbolicFunction), the integral is computed
    analytically via SymPy.  The restriction depends on both the field
    and the mesh, not either alone — neither a field nor a mesh in isolation
    carries the cell-average values.

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

        degree == ndim restricts a ZeroForm to cell averages (n-chains).
        degree == ndim - 1 restricts a OneForm's normal component to face
        integrals ((n-1)-chains).
        """


__all__ = ["RestrictionOperator"]
