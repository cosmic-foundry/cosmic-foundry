"""Simulation geometry: bounded domains composed with manifolds from theory/."""

from __future__ import annotations

from cosmic_foundry.geometry.domain import Domain
from cosmic_foundry.theory.euclidean_space import EuclideanSpace
from cosmic_foundry.theory.minkowski_space import MinkowskiSpace

__all__ = [
    "Domain",
    "EuclideanSpace",
    "MinkowskiSpace",
]
