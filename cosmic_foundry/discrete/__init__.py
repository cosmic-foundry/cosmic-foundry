"""Discrete layer: scheme description on finite index sets.

Boundary rule: this package may not import from any third-party package.
Enforced by tests/test_theory_no_third_party_imports.py alongside
foundation/ and continuous/.
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
