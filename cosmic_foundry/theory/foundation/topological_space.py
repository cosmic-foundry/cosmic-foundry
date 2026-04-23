"""TopologicalSpace ABC."""

from __future__ import annotations

from cosmic_foundry.theory.foundation.set import Set


class TopologicalSpace(Set):  # noqa: B024
    """A set equipped with a topology: a distinguished family of open subsets
    of S satisfying the axioms of a topological space.

    The topology determines which maps into and out of S are continuous,
    which sequences converge, and which subsets are compact or connected.
    It carries no metric, no notion of dimension, and no smooth structure —
    those are added by subclasses.

    Subclasses add structure:
    - Topological space that is locally Euclidean — TopologicalManifold
    """


__all__ = ["TopologicalSpace"]
