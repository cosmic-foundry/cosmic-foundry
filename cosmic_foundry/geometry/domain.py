"""Domain — a manifold equipped with physical bounds."""

from __future__ import annotations

from dataclasses import dataclass

from cosmic_foundry.theory.smooth_manifold import SmoothManifold


@dataclass(frozen=True)
class Domain:
    """A SmoothManifold with physical origin and size.

    Domain is the continuous description of a simulation region: which
    manifold it lives on (encoding dimension and geometry) and where it
    sits in physical space.  It replaces the raw keyword arguments
    previously passed to PartitionDomain.execute.

    Required:
        manifold — the manifold the domain lives on; ndim is derived from it
        origin   — physical coordinate of the domain's lower corner,
                   one value per axis
        size     — extent of the domain along each axis; all entries > 0
    """

    manifold: SmoothManifold
    origin: tuple[float, ...]
    size: tuple[float, ...]

    def __post_init__(self) -> None:
        n = self.manifold.ndim
        if len(self.origin) != n:
            raise ValueError(
                f"origin has {len(self.origin)} entries but manifold.ndim = {n}"
            )
        if len(self.size) != n:
            raise ValueError(
                f"size has {len(self.size)} entries but manifold.ndim = {n}"
            )
        if any(s <= 0 for s in self.size):
            raise ValueError(f"all size entries must be positive, got {self.size}")


__all__ = ["Domain"]
