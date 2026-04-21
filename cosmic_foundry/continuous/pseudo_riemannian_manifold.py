"""PseudoRiemannianManifold and MetricTensor ABCs."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.continuous.field import SymmetricTensorField
from cosmic_foundry.continuous.manifold import Manifold


class PseudoRiemannianManifold(Manifold):
    """A Manifold equipped with a non-degenerate metric tensor of
    indefinite signature.

    A pseudo-Riemannian manifold (M, g) adds a smoothly-varying symmetric
    bilinear form g: TₓM × TₓM → ℝ at each point x ∈ M.  The form g is
    non-degenerate but not necessarily positive-definite; its signature
    (p, q) records how many eigenvalues are positive (p) and how many are
    negative (q), with p + q = ndim.

    General relativistic spacetimes are pseudo-Riemannian with Lorentzian
    signature (1, 3) or (3, 1) depending on convention.

    Required:
        signature — metric signature as (p, q); ndim is derived as p + q
        metric    — the metric tensor g equipping this manifold with geometry
    """

    @property
    @abstractmethod
    def signature(self) -> tuple[int, int]:
        """Metric signature (p, q): p positive, q negative eigenvalues."""

    @property
    @abstractmethod
    def metric(self) -> MetricTensor:
        """The metric tensor g that equips this manifold with its geometry."""

    @property
    def ndim(self) -> int:
        """Topological dimension, derived from metric signature as p + q."""
        return sum(self.signature)


class MetricTensor(SymmetricTensorField):  # noqa: B024
    """The metric tensor g on a pseudo-Riemannian manifold (M, g).

    g is a smoothly-varying non-degenerate symmetric (0,2)-tensor field.
    Its signature (p, q) is inherited from the manifold.  Concrete
    subclasses are responsible for the non-degeneracy condition and, on
    Riemannian manifolds, positive-definiteness; neither can be enforced
    at the ABC level.

    The manifold narrows to PseudoRiemannianManifold, because a metric tensor
    is precisely the additional structure that makes a manifold pseudo-Riemannian.

    Required:
        manifold — the pseudo-Riemannian manifold this metric is defined on
    """

    @property
    @abstractmethod
    def manifold(self) -> PseudoRiemannianManifold:
        """The pseudo-Riemannian manifold on which this metric is defined."""


__all__ = ["MetricTensor", "PseudoRiemannianManifold"]
