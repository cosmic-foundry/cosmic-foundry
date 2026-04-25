"""NumericalFlux ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod


class NumericalFlux(ABC):
    """Approximates the face-averaged flux F·n̂·|face_area| from cell averages.

    A NumericalFlux maps cell-average values on a mesh and a face description
    to a scalar approximation of the face-integrated normal flux
    F·n̂·|face_area|.  The approximation order p is the composite convergence
    order:

        order = min(reconstruction_order, face_quadrature_order,
                    deconvolution_order)

    Each component must independently achieve order p; the class is
    responsible for ensuring they do.  For Lanes B and C, the composite
    order is verified algebraically via SymPy Taylor expansion in
    tests/test_convergence_order.py.

    Earns its class by: order is a verifiable integer claim — the Lane C
    Taylor expansion of the composite face flux against the exact
    face-averaged flux yields leading error O(hᵖ), where p = order.

    Required:
        order — composite convergence order
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """Composite convergence order of the scheme."""


__all__ = ["NumericalFlux"]
