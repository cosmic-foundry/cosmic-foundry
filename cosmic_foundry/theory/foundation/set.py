"""Set ABC."""

from __future__ import annotations

from abc import ABC


class Set(ABC):  # noqa: B024
    """Abstract base for all set types: the collection S of elements on which
    computation is defined.

    A Set is the most primitive mathematical concept in the hierarchy — it is
    any well-defined collection of objects.  It carries no geometry, no metric,
    no notion of dimension, and no notion of order.  Every more specific
    concept (manifold, indexed set, discretization) is a Set with additional
    structure.

    A Set is not a region, a grid, or an extent — it is the abstract type
    that unifies every input space over which fields and functions can be
    defined.  Concretely: a Riemannian manifold is a Set with a smooth
    structure and a metric; a finite index set is a Set with a bijection to
    a subset of ℤⁿ.
    """


__all__ = [
    "Set",
]
