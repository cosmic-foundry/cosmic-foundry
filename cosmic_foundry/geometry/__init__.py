"""Concrete geometry objects for Cosmic Foundry simulations.

geometry/ is the concreteness layer — instantiable coordinate geometry
infrastructure with no physics commitment.  It depends on theory/ for
abstract interfaces and contributes nothing back to that layer.

Exported objects
----------------
CartesianChart  — identity coordinate chart on EuclideanSpace
CartesianMesh   — uniform Cartesian mesh on flat Euclidean space
EuclideanSpace  — flat Euclidean ℝⁿ with g = I
"""

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.euclidean_space import CartesianChart, EuclideanSpace

__all__ = ["CartesianChart", "CartesianMesh", "EuclideanSpace"]
