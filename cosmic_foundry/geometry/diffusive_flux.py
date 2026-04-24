"""DiffusiveFlux: NumericalFlux for diffusion equations F(U) = -∇U."""

from __future__ import annotations

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


def _derive_stencil_coeffs(order: int) -> tuple[sympy.Rational, ...]:
    """Derive antisymmetric cell-average FD coefficients for φ'(face) at order p.

    Solves for the n = p/2 positive-offset coefficients c_0, ..., c_{n-1}
    corresponding to face offsets +1/2, +3/2, ..., +(2n-1)/2 in units of h.
    The full stencil follows by antisymmetry: c(−ξ) = −c(+ξ).

    Conditions: for each odd moment k ∈ {1, 3, ..., 2n−1},
        Σ_{j=0}^{n-1} 2 c_j · ∫_{ξⱼ − 1/2}^{ξⱼ + 1/2} t^k dt  =  δ_{k,1}

    where ξⱼ = j + 1/2 is the j-th positive offset and the integral is the
    cell-average moment of t^k (in units of h).  The δ_{k,1} condition pins
    the first derivative; all higher odd moments vanish to achieve order p.

    Returns the coefficients as exact sympy.Rational values.
    """
    n = order // 2
    offsets = [sympy.Rational(2 * j + 1, 2) for j in range(n)]
    c = sympy.symbols(f"c:{n}")
    t = sympy.Symbol("t")

    def cell_avg_moment(k: int, xi: sympy.Rational) -> sympy.Expr:
        return sympy.integrate(
            t**k, (t, xi - sympy.Rational(1, 2), xi + sympy.Rational(1, 2))
        )

    eqs = [
        sympy.Eq(
            sum(2 * c[j] * cell_avg_moment(k, offsets[j]) for j in range(n)),
            1 if k == 1 else 0,
        )
        for k in range(1, 2 * n, 2)
    ]
    sol = sympy.solve(eqs, c)
    return tuple(sol[ci] for ci in c)


class DiffusiveFlux(NumericalFlux):
    """Numerical flux for the diffusive flux F(φ) = -∇φ.

    DiffusiveFlux approximates the face-averaged normal flux -∂φ/∂xₐ·|Aₐ|
    at the interface between two adjacent cells along axis a, where |Aₐ| is
    the face area perpendicular to axis a.

    One class, many instances: DiffusiveFlux(order) for any even order ≥ 2.
    DiffusiveFlux(2) and DiffusiveFlux(4) are parameterized instances, not
    subclasses.  Both satisfy the same Lane C contract at their respective
    orders (see tests/test_diffusive_flux.py).

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
        self._coeffs: tuple[sympy.Rational, ...] = _derive_stencil_coeffs(order)

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
