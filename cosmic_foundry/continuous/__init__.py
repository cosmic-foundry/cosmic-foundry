"""Continuous layer: manifolds, fields, operators, boundary conditions.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.continuous.atlas import Atlas
from cosmic_foundry.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.continuous.chart import Chart
from cosmic_foundry.continuous.constraint import Constraint
from cosmic_foundry.continuous.differential_form import (
    CovectorField,
    DifferentialForm,
    ScalarField,
)
from cosmic_foundry.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.continuous.field import (
    Field,
    SymmetricTensorField,
    TensorField,
    VectorField,
)
from cosmic_foundry.continuous.local_boundary_condition import LocalBoundaryCondition
from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.continuous.metric_tensor import MetricTensor
from cosmic_foundry.continuous.non_local_boundary_condition import (
    NonLocalBoundaryCondition,
)
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)

__all__ = [
    "Atlas",
    "BoundaryCondition",
    "Chart",
    "Constraint",
    "CovectorField",
    "DifferentialForm",
    "DifferentialOperator",
    "Field",
    "LocalBoundaryCondition",
    "Manifold",
    "MetricTensor",
    "NonLocalBoundaryCondition",
    "PseudoRiemannianManifold",
    "ScalarField",
    "SymmetricTensorField",
    "TensorField",
    "VectorField",
]
