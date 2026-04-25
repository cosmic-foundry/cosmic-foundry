"""Concrete geometry objects for Cosmic Foundry simulations.

geometry/ is the concreteness layer — instantiable coordinate geometry
infrastructure with no physics commitment.  It depends on theory/ for
abstract interfaces and contributes nothing back to that layer.

Exported objects
----------------
AdvectiveFlux                — NumericalFlux for F(φ) = v·φ; order ∈ {2, 4}
CartesianChart               — identity coordinate chart on EuclideanManifold
CartesianMesh                — uniform Cartesian mesh on flat Euclidean space
CartesianRestrictionOperator — analytic cell-average restriction on CartesianMesh
DiffusiveFlux                — NumericalFlux for F(φ) = -∇φ; order ∈ {2, 4}
EuclideanManifold            — flat Euclidean ℝⁿ with g = I
FVMDiscretization            — FVM assembler: DivergenceFormEquation → DiscreteOperator
SchwarzschildManifold        — Schwarzschild vacuum solution; static spherically
                               symmetric spacetime with Lorentzian signature
"""

from cosmic_foundry.geometry.advective_flux import AdvectiveFlux
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianRestrictionOperator,
)
from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from cosmic_foundry.geometry.euclidean_manifold import CartesianChart, EuclideanManifold
from cosmic_foundry.geometry.fvm_discretization import FVMDiscretization
from cosmic_foundry.geometry.schwarzschild_manifold import SchwarzschildManifold

__all__ = [
    "AdvectiveFlux",
    "CartesianChart",
    "CartesianMesh",
    "CartesianRestrictionOperator",
    "DiffusiveFlux",
    "EuclideanManifold",
    "FVMDiscretization",
    "SchwarzschildManifold",
]
