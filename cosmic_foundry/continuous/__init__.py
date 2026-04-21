"""Continuous layer: manifolds, fields, operators, boundary conditions.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.continuous.constraint import Constraint
from cosmic_foundry.continuous.differential_form import DifferentialForm
from cosmic_foundry.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.continuous.field import (
    Field,
    SymmetricTensorField,
    TensorField,
)
from cosmic_foundry.continuous.local_boundary_condition import LocalBoundaryCondition
from cosmic_foundry.continuous.manifold import Atlas, Manifold
from cosmic_foundry.continuous.metric_tensor import MetricTensor
from cosmic_foundry.continuous.non_local_boundary_condition import (
    NonLocalBoundaryCondition,
)
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)
from cosmic_foundry.continuous.topological_manifold import TopologicalManifold

__all__ = [
    "Atlas",
    "BoundaryCondition",
    "Constraint",
    "DifferentialForm",
    "DifferentialOperator",
    "Field",
    "LocalBoundaryCondition",
    "Manifold",
    "MetricTensor",
    "NonLocalBoundaryCondition",
    "PseudoRiemannianManifold",
    "SymmetricTensorField",
    "TensorField",
    "TopologicalManifold",
]
