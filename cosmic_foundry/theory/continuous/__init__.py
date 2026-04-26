"""Continuous layer: manifolds, fields, operators, boundary conditions.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.theory.continuous.advection_diffusion_operator import (
    AdvectionDiffusionOperator,
)
from cosmic_foundry.theory.continuous.advection_operator import AdvectionOperator
from cosmic_foundry.theory.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.theory.continuous.constraint import Constraint
from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ThreeForm,
    TwoForm,
    ZeroForm,
)
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.diffusion_operator import DiffusionOperator
from cosmic_foundry.theory.continuous.dirichlet_bc import DirichletBC
from cosmic_foundry.theory.continuous.divergence_form_equation import (
    DivergenceFormEquation,
)
from cosmic_foundry.theory.continuous.exterior_derivative import ExteriorDerivative
from cosmic_foundry.theory.continuous.field import (
    Field,
    SymmetricTensorField,
    TensorField,
)
from cosmic_foundry.theory.continuous.local_boundary_condition import (
    LocalBoundaryCondition,
)
from cosmic_foundry.theory.continuous.manifold import (
    Atlas,
    Chart,
    Diffeomorphism,
    Manifold,
    Point,
)
from cosmic_foundry.theory.continuous.non_local_boundary_condition import (
    NonLocalBoundaryCondition,
)
from cosmic_foundry.theory.continuous.periodic_bc import PeriodicBC
from cosmic_foundry.theory.continuous.poisson_equation import PoissonEquation
from cosmic_foundry.theory.continuous.pseudo_riemannian_manifold import (
    MetricTensor,
    PseudoRiemannianManifold,
    RiemannianManifold,
)
from cosmic_foundry.theory.continuous.symbolic_function import SymbolicFunction
from cosmic_foundry.theory.continuous.topological_manifold import TopologicalManifold

__all__ = [
    "AdvectionDiffusionOperator",
    "AdvectionOperator",
    "Atlas",
    "BoundaryCondition",
    "Chart",
    "DirichletBC",
    "DiffusionOperator",
    "Constraint",
    "Diffeomorphism",
    "DifferentialForm",
    "DifferentialOperator",
    "DivergenceFormEquation",
    "ExteriorDerivative",
    "Field",
    "LocalBoundaryCondition",
    "Manifold",
    "MetricTensor",
    "NonLocalBoundaryCondition",
    "OneForm",
    "PeriodicBC",
    "Point",
    "PoissonEquation",
    "PseudoRiemannianManifold",
    "RiemannianManifold",
    "SymbolicFunction",
    "SymmetricTensorField",
    "TensorField",
    "ThreeForm",
    "TopologicalManifold",
    "TwoForm",
    "ZeroForm",
]
