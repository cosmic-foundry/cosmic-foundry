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

    One class, many instances: DiffusiveFlux(order) for any even order ≥ 2.
    DiffusiveFlux(2) and DiffusiveFlux(4) are parameterized instances, not
    subclasses.  Both satisfy the same Lane C contract at their respective
    orders (see tests/test_convergence_order.py).

    Stencil derivation (Lane C):

    At order p = 2n, the stencil uses 2n cells (n on each side of the face).
    Coefficients c_0, ..., c_{n-1} are derived symbolically at construction
    by solving the antisymmetric cell-average moment system:

        gradient ≈ (1/h) Σ_{k=0}^{n-1} c_k · (φ̄_{i+k} − φ̄_{i−1−k})

    where c_k is the coefficient for face offsets ±(2k+1)/2.  The unique
    solution kills all odd Taylor error terms through order h^{p-1}, giving
    leading error O(h^p).

    Explicit results: p=2 → (1,); p=4 → (5/4, -1/12).  These match the
    hardcoded stencils traditionally written as (1,-1)/h and
    (1,-15,15,-1)/(12h) respectively.

    Parameters
    ----------
    order:
        Even integer ≥ 2.  The stencil coefficients are computed via SymPy
        at construction time (~10–40 ms); __call__ uses only arithmetic.

    __call__ signature:
        (U, mesh, axis, idx_low) -> F·n̂·|face_area|

        U        — cell averages (MeshFunction callable with cell index)
        mesh     — CartesianMesh providing spacing and face_area
        axis     — normal axis a ∈ [0, ndim)
        idx_low  — index of the cell on the low side of the face;
                   the high cell is at idx_low with idx_low[axis] + 1.
                   Caller must ensure cells idx_low[axis] − (n−1) through
                   idx_low[axis] + n exist (n = order // 2).
    """

    def __init__(self, order: int) -> None:
        if order < 2 or order % 2 != 0:
            raise ValueError(f"DiffusiveFlux order must be even and ≥ 2; got {order}")
        self._order = order

        # Derive stencil coefficients from first principles.
        #
        # Goal: find c_0, ..., c_{n-1} (n = order//2) such that the
        # antisymmetric stencil
        #
        #   gradient ≈ (1/h) Σ_{k=0}^{n-1} c_k (φ̄_{+ξ_k} − φ̄_{-ξ_k})
        #
        # approximates φ'(face) to O(hᵖ), where ξ_k = (2k+1)/2 are the
        # positive cell-center offsets from the face in units of h.
        #
        # Cell-average Taylor expansion (in units of h):
        #
        #   φ̄_ξ = ∫_{ξ-1/2}^{ξ+1/2} φ(face + sh) ds
        #        = Σ_{m≥0} (φ^(m)(face) / m!) h^m · ∫_{ξ-1/2}^{ξ+1/2} s^m ds
        #
        # Denote M_m(ξ) = ∫_{ξ-1/2}^{ξ+1/2} s^m ds (the m-th cell-average moment).
        #
        # The antisymmetric difference φ̄_{+ξ} − φ̄_{-ξ} retains only odd m
        # (M_m(−ξ) = (−1)^m M_m(ξ), so even m cancels).  The stencil sum becomes
        #
        #   (1/h) Σ_k c_k (φ̄_{+ξ_k} − φ̄_{-ξ_k})
        #   = Σ_{m odd} (φ^(m)(face) / m!) h^{m-1} · 2 Σ_k c_k M_m(ξ_k)
        #
        # Matching to φ'(face) requires:
        #   m=1: 2 Σ_k c_k M_1(ξ_k) = 1        (pin the first derivative)
        #   m=3,5,...,2n-1: 2 Σ_k c_k M_m(ξ_k) = 0  (kill error through h^{p-1})
        #
        # These n conditions in n unknowns have a unique solution; solving them
        # yields O(hᵖ) accuracy.  (Even m vanish by antisymmetry; m ≥ 2n+1 give
        # error O(hᵖ) after dividing by h.)

        n = order // 2
        s = sympy.Symbol("s")
        offsets = [sympy.Rational(2 * k + 1, 2) for k in range(n)]
        c = sympy.symbols(f"c:{n}")

        moments = [
            [
                sympy.integrate(
                    s**m, (s, xi - sympy.Rational(1, 2), xi + sympy.Rational(1, 2))
                )
                for xi in offsets
            ]
            for m in range(1, 2 * n, 2)
        ]

        eqs = [
            sympy.Eq(
                2 * sum(c[k] * moments[i][k] for k in range(n)),
                1 if i == 0 else 0,
            )
            for i in range(n)
        ]

        sol = sympy.solve(eqs, c)
        self._coeffs: tuple[sympy.Rational, ...] = tuple(sol[ci] for ci in c)

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

        The high-side neighbor is idx_low with idx_low[axis] incremented by 1.
        The stencil width is order // 2 cells on each side of the face.
        """
        h: sympy.Expr = mesh._spacing[axis]
        face_area: sympy.Expr = mesh.face_area(axis)

        def shift(idx: tuple[int, ...], delta: int) -> tuple[int, ...]:
            return idx[:axis] + (idx[axis] + delta,) + idx[axis + 1 :]

        gradient = (
            sum(
                c_k * (U(shift(idx_low, k + 1)) - U(shift(idx_low, -k)))  # type: ignore[arg-type]
                for k, c_k in enumerate(self._coeffs)
            )
            / h
        )

        return sympy.Rational(-1) * gradient * face_area


__all__ = ["DiffusiveFlux"]
