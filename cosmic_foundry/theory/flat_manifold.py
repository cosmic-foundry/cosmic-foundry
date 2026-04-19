"""FlatManifold ABC."""

from __future__ import annotations

from cosmic_foundry.theory.pseudo_riemannian_manifold import PseudoRiemannianManifold


class FlatManifold(PseudoRiemannianManifold):
    """A PseudoRiemannianManifold with zero Riemann curvature tensor.

    Flatness means the Riemann tensor R^ρ_σμν vanishes everywhere, so
    parallel transport is path-independent and global Cartesian coordinates
    exist.  The metric signature is not constrained here — both positive-
    definite (Euclidean) and Lorentzian signatures are flat manifolds.

    Branches from PseudoRiemannianManifold rather than RiemannianManifold so
    that EuclideanSpace (signature (n, 0)) and MinkowskiSpace (signature
    (1, 3)) can both inherit from this class.

    Required (inherited):
        signature — metric signature (p, q); ndim = p + q
    """


__all__ = [
    "FlatManifold",
]
