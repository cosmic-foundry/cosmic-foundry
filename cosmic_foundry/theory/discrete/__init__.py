"""Discrete layer: mesh structure and operator derivations.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.theory.discrete.cell_complex import CellComplex
from cosmic_foundry.theory.discrete.cell_field import CellField
from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.face_field import FaceField
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux
from cosmic_foundry.theory.discrete.restriction_operator import RestrictionOperator
from cosmic_foundry.theory.discrete.structured_mesh import StructuredMesh

__all__ = [
    "CellComplex",
    "CellField",
    "DiscreteField",
    "DiscreteOperator",
    "Discretization",
    "FaceField",
    "Mesh",
    "NumericalFlux",
    "RestrictionOperator",
    "StructuredMesh",
]
