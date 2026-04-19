"""Pure mathematical theory: ABCs for sets, manifolds, discretizations, functions."""

from __future__ import annotations

from cosmic_foundry.theory.discretization import Discretization
from cosmic_foundry.theory.field import ContinuousField, Field, ScalarField, TensorField
from cosmic_foundry.theory.function import Function
from cosmic_foundry.theory.indexed_set import IndexedSet
from cosmic_foundry.theory.located_discretization import LocatedDiscretization
from cosmic_foundry.theory.modal_discretization import ModalDiscretization
from cosmic_foundry.theory.pseudo_riemannian_manifold import PseudoRiemannianManifold
from cosmic_foundry.theory.riemannian_manifold import RiemannianManifold
from cosmic_foundry.theory.set import Set
from cosmic_foundry.theory.smooth_manifold import SmoothManifold

__all__ = [
    "ContinuousField",
    "Discretization",
    "Field",
    "Function",
    "IndexedSet",
    "LocatedDiscretization",
    "ModalDiscretization",
    "PseudoRiemannianManifold",
    "RiemannianManifold",
    "ScalarField",
    "Set",
    "SmoothManifold",
    "TensorField",
]
