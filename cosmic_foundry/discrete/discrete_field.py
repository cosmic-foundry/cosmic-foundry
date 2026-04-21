"""DiscreteField hierarchy: f: I → V on a finite index set."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from cosmic_foundry.continuous.differential_form import DifferentialForm
from cosmic_foundry.continuous.field import Field, TensorField
from cosmic_foundry.foundation.function import Function
from cosmic_foundry.foundation.indexed_set import IndexedSet

V = TypeVar("V")  # Value type


class DiscreteField(Function[IndexedSet, Any], Generic[V]):
    """A field defined on a finite index set: f: I → V.

    A DiscreteField assigns a value in V to every element of an IndexedSet.
    It is the discrete counterpart to Field, but is not a subtype of it:
    its domain is an IndexedSet rather than a Manifold.  Both inherit from
    Function in foundation/.

    When approximates is not None, this field is a finite approximation of
    the named continuous field.  When approximates is None, the discrete
    field is a primary mathematical object — the data IS the object, with
    no continuous antecedent.

    Required:
        grid — the IndexedSet (grid) on which this field is defined
    """

    @property
    @abstractmethod
    def grid(self) -> IndexedSet:
        """The index set on which this field is defined."""

    @property
    def approximates(self) -> Field | None:
        """The continuous Field this discrete field approximates, or None."""
        return None


class DiscreteScalarField(DiscreteField[float]):
    """A discrete scalar field: one real value per grid cell.

    The discrete counterpart to a degree-0 DifferentialForm (scalar field).
    approximates narrows to Optional[DifferentialForm] so that convergence
    checks are typed correctly.
    """

    @property
    def approximates(self) -> DifferentialForm | None:
        """The continuous DifferentialForm this field approximates, or None."""
        return None


class DiscreteVectorField(DiscreteField[Any]):
    """A discrete vector field: one vector per grid cell.

    The discrete counterpart to a (1, 0) TensorField (vector field).
    approximates narrows to Optional[TensorField].
    """

    @property
    def approximates(self) -> TensorField | None:
        """The continuous TensorField this field approximates, or None."""
        return None


__all__ = [
    "DiscreteField",
    "DiscreteScalarField",
    "DiscreteVectorField",
]
