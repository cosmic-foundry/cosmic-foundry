"""FaceField: abstract DiscreteField on mesh faces."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.mesh import Mesh

_V = TypeVar("_V")


class FaceField(DiscreteField[_V]):
    """Abstract face-indexed DiscreteField: values defined at cell faces.

    A face is identified by (axis, idx_low): axis ∈ [0, ndim) is the normal
    direction and idx_low ∈ ℤⁿ is the multi-index of the low-side cell.  The
    high-side cell is at idx_low with coordinate axis incremented by one.

    FaceField is the discrete counterpart of differential forms:
        FaceField[scalar]        — 1-form; scalar flux F·n̂·|A| through each face;
                                   discrete analog of OneForm
        FaceField[sympy.Matrix]  — 2-form; matrix-valued face flux (e.g. stress
                                   tensor); discrete analog of TwoForm

    FaceField is the canonical return type of NumericalFlux.__call__ and
    CartesianRestrictionOperator (degree = ndim−1).

    Required (in addition to DiscreteField.mesh):
        __call__ — evaluate the field at a face index (axis, idx_low)

    Concrete subclasses:
        _CallableFaceField (this module) — callable-backed implementation
    """

    @abstractmethod
    def __call__(self, face: tuple[int, tuple[int, ...]]) -> _V:  # type: ignore[override]
        """Evaluate the field at face (axis, idx_low)."""


class _CallableFaceField(FaceField[_V]):
    """Callable-backed concrete FaceField.

    Stores a callable fn: (axis, idx_low) → V and evaluates it on demand.
    This is the standard concrete FaceField used by NumericalFlux implementations
    and CartesianRestrictionOperator; the callable is evaluated only at the face
    index the caller requests.

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
