"""Chart ABC: a local coordinate system on a manifold."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from cosmic_foundry.foundation.function import Function

if TYPE_CHECKING:
    from cosmic_foundry.continuous.manifold import Manifold

D = TypeVar("D")  # type of domain points
C = TypeVar("C")  # type of codomain coordinates


class Chart(Function[D, C], Generic[D, C]):
    """A local coordinate system on a manifold: a diffeomorphism φ: U → V.

    A chart assigns coordinates to points in an open subset U of a manifold
    M by mapping them homeomorphically onto an open subset V of ℝⁿ.
    The component functions x¹, …, xⁿ of φ are the local coordinates.

    A chart is a diffeomorphism: smooth, injective, and with a smooth inverse.
    The dimension of the codomain equals the dimension of the domain.

    Required:
        domain   — the open subset U ⊂ M where this chart is defined
        codomain — the open subset V ⊂ ℝⁿ represented as a Manifold
        inverse  — the smooth inverse φ⁻¹: V → U
    """

    @property
    @abstractmethod
    def domain(self) -> Manifold:
        """The open subset of M on which this chart is defined."""

    @property
    @abstractmethod
    def codomain(self) -> Manifold:
        """The open subset of ℝⁿ that is the image of this chart."""

    @property
    @abstractmethod
    def inverse(self) -> Function[Any, Any]:
        """The smooth inverse φ⁻¹: V → U."""


__all__ = ["Chart"]
