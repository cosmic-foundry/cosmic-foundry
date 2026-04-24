"""Field hierarchy: f: M → V.

Field               — any assignment of values to manifold points
TensorField         — codomain is a tensor bundle T^(p,q)M
SymmetricTensorField — symmetric covariant 2-tensor; g_{ij} = g_{ji}
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

from cosmic_foundry.theory.continuous.symbolic_function import SymbolicFunction

D = TypeVar("D")  # Domain manifold type
C = TypeVar("C")  # Codomain value type


class Field(SymbolicFunction[D, C], Generic[D, C]):
    """Abstract base for all fields: f: M → V.

    A field assigns a value in V to every point in a manifold M.  D is the
    manifold type (e.g. EuclideanManifold); C is the value type (e.g.
    sympy.Expr for scalar fields).

    Evaluation: field(Point(manifold=m, chart=c, coords=(...))) → C.
    The chart check and substitution are inherited from SymbolicFunction.__call__.
    """

    @property
    @abstractmethod
    def manifold(self) -> D:
        """The manifold on which this field is defined."""


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
