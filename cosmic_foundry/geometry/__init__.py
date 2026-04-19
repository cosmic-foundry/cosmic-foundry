"""Concrete simulation geometries: flat manifolds for Newtonian and SR domains."""

from __future__ import annotations

from cosmic_foundry.geometry.domain import Domain
from cosmic_foundry.geometry.euclidean_space import EuclideanSpace
from cosmic_foundry.geometry.minkowski_space import MinkowskiSpace

__all__ = [
    "Domain",
    "EuclideanSpace",
    "MinkowskiSpace",
]
