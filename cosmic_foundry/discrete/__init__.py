"""Discrete layer: mesh structure and operator derivations.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.discrete.cell_complex import CellComplex
from cosmic_foundry.discrete.mesh import Mesh
from cosmic_foundry.discrete.mesh_function import MeshFunction
from cosmic_foundry.discrete.restriction_operator import RestrictionOperator
from cosmic_foundry.discrete.structured_mesh import StructuredMesh

__all__ = [
    "CellComplex",
    "Mesh",
    "MeshFunction",
    "RestrictionOperator",
    "StructuredMesh",
]
