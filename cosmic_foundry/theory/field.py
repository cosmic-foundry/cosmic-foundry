"""Field hierarchy: f: M → V.

- ``Field``       — abstract base; Function[Point, Value].
- ``ScalarField`` — marker for Value = ℝ.
- ``TensorField`` — abstract for Value = T^(p,q)M; carries ``tensor_type``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.function import Function

D = TypeVar("D")  # Domain (point in manifold)
C = TypeVar("C")  # Codomain (value type)


class Field(Function[D, C]):
    """Abstract base for all fields: f: M → V.

    A field assigns a value in V to every point in a manifold M.
    Subclasses specialize by the codomain V (scalar, tensor) and by
    how f is represented (analytic callable, discrete array, modal
    coefficients).

    ``name`` is required on all concrete subclasses (carried as a
    frozen-dataclass field) and identifies the physical quantity.
    """

    name: str


class ScalarField(Field):  # noqa: B024
    """A field with codomain V = ℝ.

    Marker subclass; ``__call__`` remains abstract from ``Function``.
    """


class TensorField(Field):
    """A field with codomain V = T^(p,q)M.

    Subclasses must declare the tensor type (p contravariant, q covariant
    indices).
    """

    @property
    @abstractmethod
    def tensor_type(self) -> tuple[int, int]:
        """Return (p, q) where p is contravariant and q is covariant rank."""


__all__ = [
    "Field",
    "ScalarField",
    "TensorField",
]
