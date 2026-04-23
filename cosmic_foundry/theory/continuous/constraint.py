"""Constraint ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cosmic_foundry.theory.continuous.manifold import Manifold


class Constraint(ABC):
    """Abstract base for all constraints on a field over a manifold.

    A Constraint declares that a field must satisfy some condition on a
    geometric locus (its support).  The support is a Manifold; for boundary
    conditions the support is the boundary ∂M, but the base class imposes no
    restriction on which manifold plays that role.

    Concrete implementations live in computation/ where JAX-backed evaluation
    is available.
    """

    @property
    @abstractmethod
    def support(self) -> Manifold:
        """The manifold on which this constraint is enforced."""


__all__ = ["Constraint"]
