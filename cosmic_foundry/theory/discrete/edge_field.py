"""EdgeField: abstract DiscreteField on mesh edges."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField

_V = TypeVar("_V")


class EdgeField(DiscreteField[_V]):
    """Abstract edge-indexed DiscreteField: values defined at mesh edges.

    An edge is identified by (axis, idx): axis ∈ [0, ndim) is the tangent
    direction and idx ∈ ℤⁿ is the multi-index of the low vertex.  This
    mirrors the FaceField convention of (normal_axis, idx_low) for faces.

    EdgeField is the discrete counterpart of OneForm: a 1-form integrates
    along each edge, yielding the circulation ∮ F·dl.  It is the natural
    DOF location for the electric field in MHD constrained-transport schemes,
    where Faraday's law ∂B/∂t = −curl(E) is d: Ω¹ → Ω² on the discrete chain.

    Required (in addition to DiscreteField.mesh):
        __call__ — evaluate the field at edge (axis, idx)
    """

    @abstractmethod
    def __call__(self, edge: tuple[int, tuple[int, ...]]) -> _V:  # type: ignore[override]
        """Evaluate the field at edge (tangent_axis, idx_low)."""


__all__ = ["EdgeField"]
