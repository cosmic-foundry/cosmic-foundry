"""PointField: abstract DiscreteField on mesh vertices."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.mesh import Mesh

_V = TypeVar("_V")


class PointField(DiscreteField[_V]):
    """Abstract cell-center-indexed DiscreteField: values defined at cell centers.

    A cell center is identified by a multi-index tuple (i₀, i₁, …, iₙ₋₁).
    On a mesh with cell shape (N₀, N₁, …), the index set is N₀ × N₁ × ….
    Cell center c sits at origin + (c + ½) * h along each axis.

    PointField is the discrete counterpart of ZeroForm: a 0-form assigns a
    value to each point in the domain.  It is the natural DOF location for
    finite-difference schemes, in contrast to VolumeField which stores total
    volume integrals (Ωⁿ / volume-form DOFs used in FVM).

    Required (in addition to DiscreteField.mesh):
        __call__ — evaluate the field at a cell-center multi-index idx
    """

    @abstractmethod
    def __call__(self, idx: tuple[int, ...]) -> _V:  # type: ignore[override]
        """Evaluate the field at cell-center multi-index idx."""


class _CallablePointField(PointField[_V]):
    """Callable-backed concrete PointField."""

    def __init__(self, mesh: Mesh, fn: Callable[[tuple[int, ...]], _V]) -> None:
        self._mesh = mesh
        self._fn = fn

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, idx: tuple[int, ...]) -> _V:  # type: ignore[override]
        return self._fn(idx)


__all__ = ["PointField"]
