"""Concrete geometry objects for Cosmic Foundry simulations.

geometry/ is the coordinate geometry layer — instantiable manifolds,
charts, and meshes defined by SymPy expressions, with no physics
commitment.  PDE-specific objects (fluxes, FVM discretization, State)
live in physics/.

Exported objects
----------------
CartesianChart               — identity coordinate chart on EuclideanManifold
CartesianExteriorDerivative  — exact discrete exterior derivative on CartesianMesh
CartesianMesh                — uniform Cartesian mesh on flat Euclidean space
CartesianRestrictionOperator — analytic cell-average restriction on CartesianMesh
EuclideanManifold            — flat Euclidean ℝⁿ with g = I
SchwarzschildManifold        — Schwarzschild vacuum solution; static spherically
                               symmetric spacetime with Lorentzian signature
"""

from cosmic_foundry.geometry.cartesian_exterior_derivative import (
    CartesianExteriorDerivative,
)
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianRestrictionOperator,
)
from cosmic_foundry.geometry.euclidean_manifold import CartesianChart, EuclideanManifold
from cosmic_foundry.geometry.schwarzschild_manifold import SchwarzschildManifold

__all__ = [
    "CartesianChart",
    "CartesianExteriorDerivative",
    "CartesianMesh",
    "CartesianRestrictionOperator",
    "EuclideanManifold",
    "SchwarzschildManifold",
]
