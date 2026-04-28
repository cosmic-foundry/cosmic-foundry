"""NumericalFlux ABC."""

from __future__ import annotations

from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.face_field import FaceField

_V = TypeVar("_V")


class NumericalFlux(DiscreteOperator[DiscreteField[_V], FaceField[_V]]):
    """Operator from cell-average DiscreteField to FaceField (Ω⁰ → Ω^{n−1}).

    Used as the constitutive operator f in the divergence-form factorization
    L = ∇·f, consumed by DivergenceFormDiscretization.  The narrowed output
    type makes the cochain shape part of the type, not just convention.

    Required (inherited from DiscreteOperator):
        order               — convergence order of the approximation
        continuous_operator — the continuous flux operator approximated
        __call__            — apply the operator: DiscreteField → FaceField
    """


__all__ = ["NumericalFlux"]
