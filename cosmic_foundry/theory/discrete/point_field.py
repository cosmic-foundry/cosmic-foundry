"""PointField: abstract DiscreteField on mesh vertices."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField

_V = TypeVar("_V")


class PointField(DiscreteField[_V]):
    """Abstract vertex-indexed DiscreteField: values defined at mesh vertices.

    A vertex is identified by a multi-index tuple (i₀, i₁, …, iₙ₋₁).  On a
    mesh with cell shape (N₀, N₁, …), the vertex index set is
    (N₀+1) × (N₁+1) × ….

    PointField is the discrete counterpart of ZeroForm: a 0-form assigns a
    value to each point (vertex) in the domain.  It is the natural DOF
    location for finite-difference schemes, in contrast to CellField which
    stores cell averages (Ω³ / volume-form DOFs used in FVM).

    Required (in addition to DiscreteField.mesh):
        __call__ — evaluate the field at a vertex multi-index idx
    """

    @abstractmethod
    def __call__(self, idx: tuple[int, ...]) -> _V:  # type: ignore[override]
        """Evaluate the field at vertex multi-index idx."""


__all__ = ["PointField"]
