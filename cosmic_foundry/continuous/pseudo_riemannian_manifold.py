"""PseudoRiemannianManifold ABC."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from cosmic_foundry.continuous.smooth_manifold import SmoothManifold

if TYPE_CHECKING:
    from cosmic_foundry.continuous.metric_tensor import MetricTensor


class PseudoRiemannianManifold(SmoothManifold):
    """A SmoothManifold equipped with a non-degenerate metric tensor of
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


__all__ = [
    "PseudoRiemannianManifold",
]
