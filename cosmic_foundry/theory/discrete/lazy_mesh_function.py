"""LazyMeshFunction: MeshFunction backed by a callable."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction

_V = TypeVar("_V")


class LazyMeshFunction(MeshFunction[_V]):
    """A MeshFunction whose values are computed on demand by a callable.

    LazyMeshFunction(mesh, fn) stores a callable fn; fn(idx) -> V is invoked
    lazily when a specific index is queried.  This is the standard return type
    for full-field NumericalFlux.__call__: the flux is evaluated only at the
    face index the caller requests, making the return value JAX-friendly (one
    JIT-compiled array operation when vectorized over all faces).

    Parameters
    ----------
    mesh:
        The mesh on which this function's values are defined.
    fn:
        Callable mapping a mesh-element index to a value V.  For face-valued
        MeshFunctions the index is a (axis, idx_low) pair; for cell-valued
        MeshFunctions it is a cell-index tuple.
    """

    def __init__(self, mesh: Mesh, fn: Callable[[Any], _V]) -> None:
        self._mesh = mesh
        self._fn = fn

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, idx: Any) -> _V:
        return self._fn(idx)


__all__ = ["LazyMeshFunction"]
