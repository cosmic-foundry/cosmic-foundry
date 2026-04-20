"""Continuous layer: manifolds, fields, operators, boundary conditions.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.continuous.atlas import Atlas
from cosmic_foundry.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.continuous.chart import Chart
from cosmic_foundry.continuous.differential_form import (
    CovectorField,
    DifferentialForm,
    ScalarField,
)
from cosmic_foundry.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.continuous.euclidean_space import EuclideanSpace
from cosmic_foundry.continuous.field import (
    Field,
    SymmetricTensorField,
    TensorField,
    VectorField,
)
from cosmic_foundry.continuous.flat_manifold import FlatManifold
from cosmic_foundry.continuous.identity_chart import IdentityChart
from cosmic_foundry.continuous.local_boundary_condition import LocalBoundaryCondition
from cosmic_foundry.continuous.manifold import Manifold
from cosmic_foundry.continuous.metric_tensor import MetricTensor
from cosmic_foundry.continuous.minkowski_space import MinkowskiSpace
from cosmic_foundry.continuous.non_local_boundary_condition import (
    NonLocalBoundaryCondition,
)
from cosmic_foundry.continuous.pseudo_riemannian_manifold import (
    PseudoRiemannianManifold,
)
from cosmic_foundry.continuous.riemannian_manifold import RiemannianManifold
from cosmic_foundry.continuous.single_chart_atlas import SingleChartAtlas
from cosmic_foundry.continuous.smooth_manifold import SmoothManifold

__all__ = [
    "Atlas",
    "BoundaryCondition",
    "Chart",
    "CovectorField",
    "DifferentialForm",
    "DifferentialOperator",
    "EuclideanSpace",
    "Field",
    "FlatManifold",
    "IdentityChart",
    "LocalBoundaryCondition",
    "Manifold",
    "MetricTensor",
    "MinkowskiSpace",
    "NonLocalBoundaryCondition",
    "PseudoRiemannianManifold",
    "RiemannianManifold",
    "ScalarField",
    "SingleChartAtlas",
    "SmoothManifold",
    "SymmetricTensorField",
    "TensorField",
    "VectorField",
]
