"""Field hierarchy: f: M → V.

Field               — any assignment of values to manifold points
TensorField         — codomain is a tensor bundle T^(p,q)M
SymmetricTensorField — symmetric covariant 2-tensor; g_{ij} = g_{ji}
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.foundation.function import Function

D = TypeVar("D")  # Domain (point in manifold)
C = TypeVar("C")  # Codomain (value type)


class Field(Function[D, C]):
    """Abstract base for all fields: f: M → V.

    A field assigns a value in V to every point in a manifold M.
    """

    @property
    @abstractmethod
    def manifold(self) -> Manifold:
        """The manifold on which this field is defined."""


class TensorField(Field):  # noqa: B024
    """A field whose codomain is a tensor bundle T^(p,q)M.

    Subclasses fix tensor_type to name specific tensor kinds;
    arbitrary (p, q) fields subclass TensorField directly.
    """

    @property
    @abstractmethod
    def manifold(self) -> Manifold:
        """The manifold on which this tensor field is defined."""

    @property
    @abstractmethod
    def tensor_type(self) -> tuple[int, int]:
        """Return (p, q): p contravariant indices, q covariant indices."""


class SymmetricTensorField(TensorField):  # noqa: B024
    """A symmetric covariant 2-tensor field: tensor type (0, 2), g_{ij} = g_{ji}.

    The symmetry condition is a mathematical requirement on concrete
    implementations; it cannot be enforced at the ABC level.

    Covers the metric tensor g, the viscous stress tensor σ, and any
    other symmetric bilinear form on TM.
    """

    @property
    def tensor_type(self) -> tuple[int, int]:
        return (0, 2)


__all__ = [
    "Field",
    "SymmetricTensorField",
    "TensorField",
]
