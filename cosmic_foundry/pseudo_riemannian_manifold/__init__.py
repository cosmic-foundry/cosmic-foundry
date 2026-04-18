"""PseudoRiemannianManifold ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.smooth_manifold import SmoothManifold


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

    Required (in addition to SmoothManifold.ndim):
        signature — metric signature as (p, q) with p + q == ndim
    """

    @property
    @abstractmethod
    def signature(self) -> tuple[int, int]:
        """Metric signature (p, q): p positive, q negative eigenvalues."""


__all__ = [
    "PseudoRiemannianManifold",
]
