"""FaceField: DiscreteField on mesh faces, indexed by (axis, cell_idx)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.mesh import Mesh

_V = TypeVar("_V")


class FaceField(DiscreteField[_V]):
    """A DiscreteField whose values are defined on mesh faces.

    A face is identified by (axis, idx_low): axis ∈ [0, ndim) is the normal
    direction and idx_low ∈ ℤⁿ is the multi-index of the low-side cell.  The
    high-side cell is at idx_low with coordinate axis incremented by one.

    FaceField is the discrete counterpart of a differential form:
        FaceField[scalar]        — 1-form; scalar flux F·n̂·|A| through each face;
                                   discrete analog of OneForm
        FaceField[sympy.Matrix]  — 2-form; matrix-valued face flux (e.g. stress
                                   tensor); discrete analog of TwoForm

    The value type V is unconstrained: sympy.Expr for symbolic evaluation
    (convergence proofs), float for numeric paths, sympy.Matrix for tensor flux.

    FaceField is the canonical return type of NumericalFlux.__call__: the flux
    callable is stored and evaluated on demand, decoupling mesh traversal from
    flux computation.

    Parameters
    ----------
    mesh:
        The mesh whose faces this field is defined on.
    fn:
        Callable mapping a face index (axis, idx_low) to a value of type V.
    """

    def __init__(
        self,
        mesh: Mesh,
        fn: Callable[[tuple[int, tuple[int, ...]]], _V],
    ) -> None:
        self._mesh = mesh
        self._fn = fn

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, face: tuple[int, tuple[int, ...]]) -> _V:  # type: ignore[override]
        return self._fn(face)


__all__ = ["FaceField"]
