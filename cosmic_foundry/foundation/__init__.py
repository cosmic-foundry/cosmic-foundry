"""Foundation: primitive mathematical abstractions shared by all layers.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.foundation.function import Function
from cosmic_foundry.foundation.homeomorphism import Homeomorphism
from cosmic_foundry.foundation.indexed_family import IndexedFamily
from cosmic_foundry.foundation.indexed_set import IndexedSet
from cosmic_foundry.foundation.set import Set
from cosmic_foundry.foundation.topological_space import TopologicalSpace

__all__ = [
    "Function",
    "Homeomorphism",
    "IndexedFamily",
    "IndexedSet",
    "Set",
    "TopologicalSpace",
]
