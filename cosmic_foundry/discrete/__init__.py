"""Discrete layer: scheme description on finite index sets.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.discrete.discrete_field import (
    DiscreteField,
    DiscreteScalarField,
    DiscreteVectorField,
)

__all__ = [
    "DiscreteField",
    "DiscreteScalarField",
    "DiscreteVectorField",
]
