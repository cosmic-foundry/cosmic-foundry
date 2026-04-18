"""Discretization ABC."""

from __future__ import annotations

from cosmic_foundry.indexed_set import IndexedSet


class Discretization(IndexedSet):  # noqa: B024
    """A finite-dimensional approximation of functions on a manifold M.

    A Discretization is an IndexedSet I together with the intent that it
    approximates some function space on a smooth manifold M.  The index set
    I provides the bookkeeping structure (ndim, shape); the approximation
    scheme determines how elements of I represent information about M.

    Two fundamentally different schemes are distinguished by subclasses:

    - LocatedDiscretization: each index i ∈ I is associated with a point
      φ(i) ∈ M.  DOFs are values at specific locations (FDM, FVM, FEM,
      particles).

    - ModalDiscretization: each index i ∈ I labels a basis function bᵢ on
      M.  DOFs are modal coefficients (spectral methods).

    A Discretization carries no metric, no geometry beyond what is needed
    to define the approximation, and no payload — payloads live in
    PatchFunction.
    """


__all__ = [
    "Discretization",
]
