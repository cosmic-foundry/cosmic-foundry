"""Domain — a ManifoldWithBoundary carving a finite region from a SmoothManifold."""

from __future__ import annotations

from dataclasses import dataclass, field

from cosmic_foundry.theory.manifold_with_boundary import ManifoldWithBoundary
from cosmic_foundry.theory.smooth_manifold import SmoothManifold


@dataclass(frozen=True, slots=True)
class Domain(ManifoldWithBoundary):
    """A finite simulation region: a SmoothManifold with physical origin and size.

    Domain is the continuous description of a simulation region — which
    manifold it lives in (encoding dimension and geometry) and where the
    region sits in physical space.  Domain IS-A ManifoldWithBoundary; its
    boundary consists of 2*ndim axis-aligned face domains of dimension ndim-1.

    Required:
        manifold — the ambient SmoothManifold; ndim is derived from it
        origin   — physical coordinate of the domain's lower corner,
                   one value per axis
        size     — extent of the domain along each axis; all entries > 0
    """

    manifold: SmoothManifold = field()
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

    @property
    def ndim(self) -> int:
        return self.manifold.ndim

    @property
    def boundary(self) -> tuple[ManifoldWithBoundary, ...]:
        """The 2*ndim axis-aligned face domains of dimension ndim-1.

        Each face is a Domain of dimension ndim-1, using the same manifold
        restricted to the face's axes.  Faces are ordered lo then hi along
        each axis: (axis=0 lo, axis=0 hi, axis=1 lo, axis=1 hi, ...).
        """
        from cosmic_foundry.theory.euclidean_space import EuclideanSpace

        faces = []
        n = self.ndim
        for axis in range(n):
            face_origin = tuple(self.origin[i] for i in range(n) if i != axis)
            face_size = tuple(self.size[i] for i in range(n) if i != axis)
            face_manifold = EuclideanSpace(n - 1) if n > 1 else EuclideanSpace(1)
            for _ in range(2):  # lo face, hi face
                faces.append(
                    Domain(
                        manifold=face_manifold,
                        origin=face_origin,
                        size=face_size,
                    )
                )
        return tuple(faces)


__all__ = ["Domain"]
