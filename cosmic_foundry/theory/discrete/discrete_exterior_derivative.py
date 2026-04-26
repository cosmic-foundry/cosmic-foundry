"""DiscreteExteriorDerivative: abstract exact chain map on discrete fields."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.mesh import Mesh


class DiscreteExteriorDerivative(ABC):
    """Abstract discrete exterior derivative d_k: (Ω^k DOF) → (Ω^{k+1} DOF).

    The discrete exterior derivative is an exact chain map — it satisfies
    d_{k+1} ∘ d_k = 0 identically, with no truncation error.  This
    distinguishes it from DiscreteOperator, which approximates a continuous
    operator to finite order.

    In the discrete de Rham complex on a Cartesian mesh:
        d₀: PointField → EdgeField    (discrete gradient)
        d₁: EdgeField  → FaceField    (discrete curl; 3-D only)
        d₂: FaceField  → VolumeField  (discrete divergence)

    Required:
        mesh   — the mesh on which the chain complex is defined
        degree — input form degree k ∈ {0, 1, 2}
        __call__ — apply d_k: (k-form DOF) → ((k+1)-form DOF)
    """

    @property
    @abstractmethod
    def mesh(self) -> Mesh:
        """The mesh on which this operator is defined."""

    @property
    @abstractmethod
    def degree(self) -> int:
        """Input form degree k (0, 1, or 2)."""

    @abstractmethod
    def __call__(self, field: DiscreteField[Any]) -> DiscreteField[Any]:
        """Apply d_k to field."""


__all__ = ["DiscreteExteriorDerivative"]
