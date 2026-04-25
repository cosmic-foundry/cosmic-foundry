"""NumericalFlux ABC."""

from __future__ import annotations

from typing import TypeVar

from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator

_V = TypeVar("_V")


class NumericalFlux(DiscreteOperator[_V]):
    """Approximates the face-averaged flux F·n̂·|face_area| from cell averages.

    A NumericalFlux is a DiscreteOperator with a specific calling convention:
    given a cell-average MeshFunction U, it returns a face-valued MeshFunction
    whose value at face (axis, idx_low) is the approximate flux F·n̂·|face_area|
    at the interface between cells idx_low and idx_low+1 along axis.

    NumericalFlux earns its place in the hierarchy by narrowing the calling
    convention: DiscreteOperator maps any MeshFunction to any MeshFunction;
    NumericalFlux commits to the cell-average → face-flux direction, which
    is the FVM face-flux assembly pattern.

    The convergence order is the composite minimum:

        order = min(reconstruction_order, face_quadrature_order,
                    deconvolution_order)

    Each component must independently achieve order p; the class is
    responsible for ensuring they do.  For Lanes B and C, the composite
    order is verified algebraically via SymPy Taylor expansion in
    tests/test_convergence_order.py.

    Required (inherited from DiscreteOperator):
        order               — composite convergence order
        continuous_operator — the continuous flux operator approximated
        __call__            — apply the operator: MeshFunction → MeshFunction
    """


__all__ = ["NumericalFlux"]
