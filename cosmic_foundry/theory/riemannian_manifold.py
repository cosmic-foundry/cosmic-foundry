"""RiemannianManifold ABC."""

from __future__ import annotations

from cosmic_foundry.theory.pseudo_riemannian_manifold import PseudoRiemannianManifold


class RiemannianManifold(PseudoRiemannianManifold):  # noqa: B024
    """A PseudoRiemannianManifold whose metric is positive-definite.

    A Riemannian manifold (M, g) specializes the pseudo-Riemannian case to
    signature (ndim, 0): all eigenvalues of g are strictly positive.  This
    makes g a true inner product at each point, giving rise to lengths,
    angles, geodesic distance, and volume forms in the familiar Euclidean
    sense.

    Newtonian spatial domains (flat ℝⁿ with Euclidean metric, spheres,
    tori, etc.) are Riemannian manifolds.  Lorentzian spacetimes are not —
    they live at PseudoRiemannianManifold.

    No additional abstract methods beyond PseudoRiemannianManifold: the
    positive-definiteness constraint is a restriction on the value of
    signature, not a new interface requirement.  Concrete subclasses must
    still implement ndim and signature (which must satisfy q == 0).
    """


__all__ = [
    "RiemannianManifold",
]
