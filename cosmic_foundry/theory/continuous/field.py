"""Field hierarchy: f: M → V.

Field               — any assignment of values to manifold points
TensorField         — codomain is a tensor bundle T^(p,q)M
SymmetricTensorField — symmetric covariant 2-tensor; g_{ij} = g_{ji}
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from cosmic_foundry.theory.foundation.symbolic_function import SymbolicFunction

if TYPE_CHECKING:
    from cosmic_foundry.theory.continuous.point import Point

D = TypeVar("D")  # Domain manifold type
C = TypeVar("C")  # Codomain value type


class Field(SymbolicFunction[D, C]):
    """Abstract base for all fields: f: M → V.

    A field assigns a value in V to every point in a manifold M.  D is the
    manifold type (e.g. EuclideanManifold); C is the value type (e.g.
    sympy.Expr for scalar fields).

    Evaluation is typed via evaluate(point: Point[D]) → C, which verifies
    that the point's chart matches the field's coordinate symbols before
    substituting.  The inherited __call__(*args) remains available for
    internal symbolic use where a typed Point is not needed.
    """

    @property
    @abstractmethod
    def manifold(self) -> D:
        """The manifold on which this field is defined."""

    def evaluate(self, point: Point[D]) -> C:
        """Evaluate this field at a typed point.

        Verifies that point.chart.symbols matches self.symbols, then
        substitutes point.coords into self.expr.  Raises ValueError on a
        chart mismatch so that cross-manifold evaluation is caught at runtime
        (and rejected by mypy at the type level via Point[D]).
        """
        if point.chart.symbols != self.symbols:
            raise ValueError(
                f"Chart mismatch: point uses chart with symbols "
                f"{point.chart.symbols}, but field expects {self.symbols}"
            )
        return self(*point.coords)  # type: ignore[no-any-return]


class TensorField(Field[D, C], Generic[D, C]):  # noqa: B024
    """A field whose codomain is a tensor bundle T^(p,q)M.

    Subclasses fix tensor_type to name specific tensor kinds;
    arbitrary (p, q) fields subclass TensorField directly.
    """

    @property
    @abstractmethod
    def tensor_type(self) -> tuple[int, int]:
        """Return (p, q): p contravariant indices, q covariant indices."""


class SymmetricTensorField(TensorField[D, C], Generic[D, C]):
    """A symmetric covariant 2-tensor field: tensor type (0, 2), g_{ij} = g_{ji}.

    Covers the metric tensor g, the viscous stress tensor σ, and any
    other symmetric bilinear form on TM.

    The symmetry condition is enforced through component: any valid
    subclass must satisfy component(i, j) == component(j, i) pointwise.

    Required:
        component — return the (i, j) scalar field component; must be symmetric
    """

    @property
    def tensor_type(self) -> tuple[int, int]:
        return (0, 2)

    @abstractmethod
    def component(self, i: int, j: int) -> Field:
        """Return the (i, j) component as a scalar Field on this manifold.

        Components are always expressed relative to a specific coordinate
        chart.  Implementations must document which chart they use; the
        returned Field's ``symbols`` attribute identifies the coordinate
        symbols of that chart.

        Implementations must satisfy component(i, j) == component(j, i)
        pointwise for all valid index pairs.
        """


__all__ = [
    "Field",
    "SymmetricTensorField",
    "TensorField",
]
