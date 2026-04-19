"""Pure mathematical theory: ABCs for sets, manifolds, discretizations, functions.

Boundary rule: this package may not import from any third-party package.
That boundary is the precise definition of the abstract-to-concrete transition:
mathematical concreteness (classes parameterized by Python primitives) is allowed
here; computational concreteness (JAX, NumPy, HDF5, or any other third-party
library) belongs in computation/, mesh/, or geometry/.
"""

from __future__ import annotations

from cosmic_foundry.theory.boundary_condition import BoundaryCondition
from cosmic_foundry.theory.discretization import Discretization
from cosmic_foundry.theory.euclidean_space import EuclideanSpace
from cosmic_foundry.theory.field import Field, ScalarField, TensorField
from cosmic_foundry.theory.flat_manifold import FlatManifold
from cosmic_foundry.theory.function import Function
from cosmic_foundry.theory.indexed_family import IndexedFamily
from cosmic_foundry.theory.indexed_set import IndexedSet
from cosmic_foundry.theory.local_boundary_condition import LocalBoundaryCondition
from cosmic_foundry.theory.located_discretization import LocatedDiscretization
from cosmic_foundry.theory.manifold import Manifold
from cosmic_foundry.theory.manifold_with_boundary import ManifoldWithBoundary
from cosmic_foundry.theory.minkowski_space import MinkowskiSpace
from cosmic_foundry.theory.modal_discretization import ModalDiscretization
from cosmic_foundry.theory.non_local_boundary_condition import NonLocalBoundaryCondition
from cosmic_foundry.theory.pseudo_riemannian_manifold import PseudoRiemannianManifold
from cosmic_foundry.theory.riemannian_manifold import RiemannianManifold
from cosmic_foundry.theory.set import Set
from cosmic_foundry.theory.smooth_manifold import SmoothManifold

__all__ = [
    "BoundaryCondition",
    "Discretization",
    "EuclideanSpace",
    "Field",
    "FlatManifold",
    "Function",
    "IndexedFamily",
    "IndexedSet",
    "LocalBoundaryCondition",
    "LocatedDiscretization",
    "Manifold",
    "ManifoldWithBoundary",
    "MinkowskiSpace",
    "ModalDiscretization",
    "NonLocalBoundaryCondition",
    "PseudoRiemannianManifold",
    "RiemannianManifold",
    "ScalarField",
    "Set",
    "SmoothManifold",
    "TensorField",
]
