"""Foundation: primitive mathematical abstractions shared by all layers.

Boundary rule: symbolic-reasoning layer. May only import from stdlib,
cosmic_foundry, or the approved symbolic-reasoning packages {sympy}.
"""

from __future__ import annotations

from cosmic_foundry.foundation.function import Function
from cosmic_foundry.foundation.indexed_family import IndexedFamily
from cosmic_foundry.foundation.indexed_set import IndexedSet
from cosmic_foundry.foundation.set import Set

__all__ = [
    "Function",
    "IndexedFamily",
    "IndexedSet",
    "Set",
]
