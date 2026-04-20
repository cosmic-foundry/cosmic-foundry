"""MinkowskiSpace — flat Lorentzian ℝ⁴."""

from __future__ import annotations

from cosmic_foundry.continuous.flat_manifold import FlatManifold


class MinkowskiSpace(FlatManifold):
    """ℝ⁴ with Lorentzian signature (1, 3).

    The flat pseudo-Riemannian background for special-relativistic
    simulations.  No free parameters — dimension and signature are fixed.

    Convention: signature (1, 3) means one positive (time) and three
    negative (space) eigenvalues, consistent with the particle-physics
    sign convention (+, −, −, −).
    """

    @property
    def signature(self) -> tuple[int, int]:
        """Lorentzian signature: one timelike, three spacelike directions."""
        return (1, 3)


__all__ = [
    "MinkowskiSpace",
]
