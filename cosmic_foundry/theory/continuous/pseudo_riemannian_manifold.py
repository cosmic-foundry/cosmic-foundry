"""PseudoRiemannianManifold and MetricTensor ABCs."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.theory.continuous.field import C, D, SymmetricTensorField
from cosmic_foundry.theory.continuous.manifold import Manifold


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


class RiemannianManifold(PseudoRiemannianManifold):  # noqa: B024
    """A Manifold equipped with a positive-definite metric tensor.

    A Riemannian manifold is a pseudo-Riemannian manifold whose metric has
    signature (n, 0): all eigenvalues are positive.  This makes the metric
    a true inner product at each point, enabling lengths, angles, and
    geodesics in the classical sense.

    ndim becomes the primitive; signature is derived as (ndim, 0).

    Required:
        ndim   — topological dimension n
        metric — the positive-definite metric tensor

    Derived:
        signature — always (ndim, 0)
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Topological dimension n."""

    @property
    def signature(self) -> tuple[int, int]:
        """Metric signature; derived as (ndim, 0)."""
        return (self.ndim, 0)


class MetricTensor(SymmetricTensorField[D, C]):  # noqa: B024
    """The metric tensor g on a pseudo-Riemannian manifold (M, g).

    g is a smoothly-varying non-degenerate symmetric (0,2)-tensor field.
    Its signature (p, q) is inherited from the manifold.  Concrete
    subclasses are responsible for the non-degeneracy condition and, on
    Riemannian manifolds, positive-definiteness; neither can be enforced
    at the ABC level.
    """


__all__ = ["MetricTensor", "PseudoRiemannianManifold", "RiemannianManifold"]
