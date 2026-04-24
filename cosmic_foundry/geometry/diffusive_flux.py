"""DiffusiveFlux: NumericalFlux for diffusion equations F(U) = -∇U."""

from __future__ import annotations

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


class DiffusiveFlux(NumericalFlux):
    """Numerical flux for the diffusive flux F(φ) = -∇φ.

    DiffusiveFlux approximates the face-averaged normal flux -∂φ/∂xₐ·|Aₐ|
    at the interface between two adjacent cells along axis a, where |Aₐ| is
    the face area perpendicular to axis a.

    One class, two instances: DiffusiveFlux(2) and DiffusiveFlux(4) are
    parameterized instances, not subclasses.  Both satisfy the same Lane C
    contract at their respective orders; the test that forces the design is
    that both instances pass the same symbolic Taylor-expansion checks (see
    tests/test_diffusive_flux.py).

    Stencil derivation (Lane C):

    p=2 — centered difference of cell averages:
        F·n̂ ≈ -(φ̄_{i+1} - φ̄_i) / h
        Taylor expansion shows leading error (h²/12)φ'''(x_{i+1/2}).

    p=4 — four-point antisymmetric stencil derived from cell-average Taylor
        expansion with antisymmetry constraints (see assert_* in tests/):
        F·n̂ ≈ -(φ̄_{i-1} - 15φ̄_i + 15φ̄_{i+1} - φ̄_{i+2}) / (12h)
        leading error O(h⁴); deconvolution O(h⁴) and face quadrature trivially
        exact in 1D (face = point) are verified in the accompanying tests.

    Parameters
    ----------
    order:
        Composite convergence order; must be 2 or 4.

    __call__ signature:
        (U, mesh, axis, idx_low) -> F·n̂·|face_area|

        U        — cell averages (MeshFunction callable with cell index)
        mesh     — CartesianMesh providing spacing and face_area
        axis     — normal axis a ∈ [0, ndim)
        idx_low  — index of the cell on the low side of the face;
                   the high cell is at idx_low with idx_low[axis] + 1.
    """

    def __init__(self, order: int) -> None:
        if order not in (2, 4):
            raise ValueError(f"DiffusiveFlux order must be 2 or 4; got {order}")
        self._order = order

    @property
    def order(self) -> int:
        return self._order

    def __call__(
        self,
        U: MeshFunction,
        mesh: CartesianMesh,
        axis: int,
        idx_low: tuple[int, ...],
    ) -> sympy.Expr:
        """Return -∂φ/∂x_axis · |face_area| at the face adjacent to idx_low.

        The neighbor is idx_low with idx_low[axis] incremented by 1 (the high side).
        For order=4, cells at idx_low[axis] - 1 and idx_low[axis] + 2 are also
        accessed; the caller is responsible for ensuring they exist.
        """
        h: sympy.Expr = mesh._spacing[axis]
        face_area: sympy.Expr = mesh.face_area(axis)

        def shift(idx: tuple[int, ...], delta: int) -> tuple[int, ...]:
            return idx[:axis] + (idx[axis] + delta,) + idx[axis + 1 :]

        idx_high = shift(idx_low, 1)

        if self._order == 2:
            gradient = (U(idx_high) - U(idx_low)) / h  # type: ignore[arg-type]
        else:
            idx_m1 = shift(idx_low, -1)
            idx_pp = shift(idx_high, 1)
            gradient = (
                U(idx_m1) - 15 * U(idx_low) + 15 * U(idx_high) - U(idx_pp)  # type: ignore[arg-type]
            ) / (12 * h)

        return sympy.Rational(-1) * gradient * face_area


__all__ = ["DiffusiveFlux"]
