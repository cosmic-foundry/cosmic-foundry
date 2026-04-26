"""VolumeField: abstract DiscreteField of volume integrals on mesh cells."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.mesh import Mesh

_V = TypeVar("_V")


class VolumeField(DiscreteField[_V]):
    """Abstract cell-indexed DiscreteField: total integrals over mesh cells.

    VolumeField narrows DiscreteField to the cell domain: each value is the
    total integral of a quantity over the cell volume, identified by a
    multi-index tuple (i₀, i₁, …, iₙ₋₁) drawn from the rectangular index
    set of the mesh.

    VolumeField is the discrete counterpart of the volume form (Ωⁿ in n
    dimensions): each value is the integral ∫_Ωᵢ f dV, an extensive quantity.
    This is the n-cochain in the discrete de Rham complex:

        PointField → EdgeField → FaceField → VolumeField
        (0-cochain)  (1-cochain)  (2-cochain)  (n-cochain)

    Required (in addition to DiscreteField.mesh):
        __call__(idx: tuple[int, ...]) → V  — evaluate at a cell multi-index

    Concrete subclasses:
        _CartesianVolumeIntegral (cartesian_restriction_operator.py)
        _CallableVolumeField     — callable-backed (this module)
    """

    @abstractmethod
    def __call__(self, idx: tuple[int, ...]) -> _V:  # type: ignore[override]
        """Evaluate the field at cell multi-index idx."""


class _CallableVolumeField(VolumeField[_V]):
    """Callable-backed concrete VolumeField."""

    def __init__(self, mesh: Mesh, fn: Callable[[tuple[int, ...]], _V]) -> None:
        self._mesh = mesh
        self._fn = fn

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, idx: tuple[int, ...]) -> _V:  # type: ignore[override]
        return self._fn(idx)


__all__ = ["VolumeField"]
