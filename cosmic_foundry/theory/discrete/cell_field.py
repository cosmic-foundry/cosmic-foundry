"""CellField: abstract DiscreteField on mesh cell centers."""

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_field import DiscreteField

_V = TypeVar("_V")


class CellField(DiscreteField[_V]):
    """Abstract cell-indexed DiscreteField: values defined at cell centers.

    CellField narrows DiscreteField to the cell-center domain: each value is
    associated with a cell identified by a multi-index tuple (i₀, i₁, …, iₙ₋₁)
    drawn from the rectangular index set of the mesh.

    CellField is the discrete counterpart of the volume form (Ω³ in 3D,
    Ωⁿ in n dimensions): each value is the average of a quantity over the
    cell volume.  This is the natural DOF location for FVM schemes.  The
    discrete counterpart of ZeroForm (Ω⁰, point values) is PointField.

    Required (in addition to DiscreteField.mesh):
        __call__(idx: tuple[int, ...]) → V  — evaluate at a cell multi-index

    Concrete subclasses:
        State            (physics/)           — Tensor-backed, float-valued
        _BasisField      (discretization.py)  — sympy unit basis for assemble()
        _GhostedField    (fvm_discretization.py) — ghost-cell-extended cell field
        _CartesianCellAverage (cartesian_restriction_operator.py)
    """

    @abstractmethod
    def __call__(self, idx: tuple[int, ...]) -> _V:  # type: ignore[override]
        """Evaluate the field at cell multi-index idx."""


__all__ = ["CellField"]
