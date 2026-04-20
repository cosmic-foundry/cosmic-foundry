"""MetricTensor: the metric on a pseudo-Riemannian manifold."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.continuous.field import SymmetricTensorField
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)


class MetricTensor(SymmetricTensorField):  # noqa: B024
    """The metric tensor g on a pseudo-Riemannian manifold (M, g).

    g is a smoothly-varying non-degenerate symmetric (0,2)-tensor field.
    Its signature (p, q) is inherited from the manifold.  Concrete
    subclasses are responsible for the non-degeneracy condition and, on
    Riemannian manifolds, positive-definiteness; neither can be enforced
    at the ABC level.

    The manifold narrows from SmoothManifold (required by SymmetricTensorField)
    to PseudoRiemannianManifold, because a metric tensor is precisely the
    additional structure that makes a smooth manifold pseudo-Riemannian.

    Required:
        manifold — the pseudo-Riemannian manifold this metric is defined on
    """

    @property
    @abstractmethod
    def manifold(self) -> PseudoRiemannianManifold:
        """The pseudo-Riemannian manifold on which this metric is defined."""


__all__ = ["MetricTensor"]
