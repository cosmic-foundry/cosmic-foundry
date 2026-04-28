"""Concrete geometry objects for Cosmic Foundry simulations.

geometry/ is the coordinate geometry layer — instantiable manifolds,
charts, and meshes defined by SymPy expressions, with no physics
commitment.  PDE-specific objects (fluxes, FVM discretization, State)
live in physics/.

Exported objects
----------------
CartesianChart               — identity coordinate chart on EuclideanManifold
CartesianEdgeRestriction     — analytic Rₕ¹: OneForm → EdgeField on CartesianMesh
CartesianExteriorDerivative  — exact discrete exterior derivative on CartesianMesh
CartesianFaceRestriction     — analytic Rₕⁿ⁻¹: OneForm → FaceField on CartesianMesh
CartesianMesh                — uniform Cartesian mesh on flat Euclidean space
CartesianPointRestriction    — analytic Rₕ⁰: ZeroForm → PointField on CartesianMesh
CartesianVolumeRestriction   — analytic Rₕⁿ: n-Form → VolumeField on CartesianMesh
EuclideanManifold            — flat Euclidean ℝⁿ with g = I
SchwarzschildManifold        — Schwarzschild vacuum solution; static spherically
                               symmetric spacetime with Lorentzian signature
"""

from cosmic_foundry.geometry.cartesian_exterior_derivative import (
    CartesianExteriorDerivative,
)
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianEdgeRestriction,
    CartesianFaceRestriction,
    CartesianPointRestriction,
    CartesianVolumeRestriction,
)
from cosmic_foundry.geometry.euclidean_manifold import CartesianChart, EuclideanManifold
from cosmic_foundry.geometry.schwarzschild_manifold import SchwarzschildManifold

__all__ = [
    "CartesianChart",
    "CartesianEdgeRestriction",
    "CartesianExteriorDerivative",
    "CartesianFaceRestriction",
    "CartesianMesh",
    "CartesianPointRestriction",
    "CartesianVolumeRestriction",
    "EuclideanManifold",
    "SchwarzschildManifold",
]
