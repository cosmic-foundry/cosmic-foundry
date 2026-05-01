"""Discrete layer: mesh structure and operator derivations.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.theory.discrete.advection_diffusion_flux import (
    AdvectionDiffusionFlux,
)
from cosmic_foundry.theory.discrete.advective_flux import AdvectiveFlux
from cosmic_foundry.theory.discrete.capabilities import (
    DiscreteOperatorCapability,
    DiscreteOperatorRegistry,
    DiscreteOperatorRequest,
    discrete_operator_capabilities,
    select_discrete_operator,
)
from cosmic_foundry.theory.discrete.cell_complex import CellComplex
from cosmic_foundry.theory.discrete.diffusive_flux import DiffusiveFlux
from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DirichletGhostCells,
    DiscreteBoundaryCondition,
    InhomogeneousDirichletGhostCells,
    NeumannGhostCells,
    PeriodicGhostCells,
    ZeroGhostCells,
)
from cosmic_foundry.theory.discrete.discrete_exterior_derivative import (
    DiscreteExteriorDerivative,
)
from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.divergence_form_discretization import (
    DivergenceFormDiscretization,
)
from cosmic_foundry.theory.discrete.edge_field import EdgeField
from cosmic_foundry.theory.discrete.face_field import FaceField
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux
from cosmic_foundry.theory.discrete.point_field import PointField
from cosmic_foundry.theory.discrete.restriction_operator import RestrictionOperator
from cosmic_foundry.theory.discrete.structured_mesh import StructuredMesh
from cosmic_foundry.theory.discrete.volume_field import VolumeField

__all__ = [
    "AdvectionDiffusionFlux",
    "AdvectiveFlux",
    "CellComplex",
    "DiffusiveFlux",
    "DirichletGhostCells",
    "DiscreteOperatorCapability",
    "discrete_operator_capabilities",
    "DiscreteOperatorRegistry",
    "DiscreteOperatorRequest",
    "DiscreteBoundaryCondition",
    "DiscreteExteriorDerivative",
    "DiscreteField",
    "DiscreteOperator",
    "Discretization",
    "DivergenceFormDiscretization",
    "EdgeField",
    "FaceField",
    "InhomogeneousDirichletGhostCells",
    "Mesh",
    "NeumannGhostCells",
    "NumericalFlux",
    "PeriodicGhostCells",
    "PointField",
    "RestrictionOperator",
    "select_discrete_operator",
    "StructuredMesh",
    "VolumeField",
    "ZeroGhostCells",
]
