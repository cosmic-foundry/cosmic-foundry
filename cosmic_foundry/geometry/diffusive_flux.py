"""DiffusiveFlux: NumericalFlux for diffusion equations F(U) = -∇U."""

from __future__ import annotations

from typing import ClassVar, cast

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.diffusion_operator import DiffusionOperator
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


class DiffusiveFlux(NumericalFlux[sympy.Expr]):
    """Numerical flux for the diffusive flux F(φ) = -∇φ.

    DiffusiveFlux approximates the face-averaged normal flux -∂φ/∂xₐ·|Aₐ|
    at the interface between two adjacent cells along axis a, where |Aₐ| is
    the face area perpendicular to axis a.

    One class, many instances: DiffusiveFlux(order, continuous_operator) for
    any valid order.  DiffusiveFlux(2, ...) and DiffusiveFlux(4, ...) are
    parameterized instances, not subclasses.  Both satisfy the same Lane C
    contract at their respective orders (see tests/test_convergence_order.py).

    Validity:
        min_order  = 2   — smallest supported order
        order_step = 2   — valid orders are min_order, min_order+step, ...
                           (antisymmetric stencil design constrains order
                           to even integers: odd error terms vanish by
                           antisymmetry; only even orders are achievable)

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
        Integer satisfying order >= min_order and
        (order - min_order) % order_step == 0.  Stencil coefficients are
        derived via SymPy at construction time (~10–40 ms); __call__ uses
        only arithmetic.
    continuous_operator:
        The DiffusionOperator (-∇: Ω⁰ → Ω¹) that this instance approximates.
        Must be a DiffusionOperator; the constructor guard enforces this.

    __call__ signature:
        (U: MeshFunction) -> MeshFunction

        U        — cell averages (MeshFunction callable with cell index)
        returns  — face-flux MeshFunction callable as result((axis, idx_low))
                   where idx_low is the low-side cell index of the face;
                   the high cell is idx_low with idx_low[axis] + 1.
                   Caller must ensure cells idx_low[axis] − (n−1) through
                   idx_low[axis] + n exist (n = order // 2).
    """

    min_order: ClassVar[int] = 2
    order_step: ClassVar[int] = 2

    def __init__(self, order: int, continuous_operator: DifferentialOperator) -> None:
        if order < self.min_order or (order - self.min_order) % self.order_step != 0:
            raise ValueError(
                f"DiffusiveFlux order must be >= {self.min_order} and satisfy "
                f"(order - {self.min_order}) % {self.order_step} == 0; got {order}"
            )
        if not isinstance(continuous_operator, DiffusionOperator):
            raise TypeError(
                f"DiffusiveFlux continuous_operator must be a DiffusionOperator; "
                f"got {type(continuous_operator).__name__}"
            )
        self._order = order
        self._continuous_operator = continuous_operator

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

    @property
    def continuous_operator(self) -> DifferentialOperator:
        return self._continuous_operator

    def __call__(
        self,
        U: MeshFunction[sympy.Expr],
    ) -> LazyMeshFunction[sympy.Expr]:
        """Return a face-flux MeshFunction over all faces.

        The returned MeshFunction is callable as result((axis, idx_low))
        where idx_low is the low-side cell index tuple.  Values are computed
        lazily on demand.  The mesh is inferred from U.mesh.
        """
        mesh = cast(CartesianMesh, U.mesh)

        def compute(face: tuple[int, tuple[int, ...]]) -> sympy.Expr:
            axis, idx_low = face
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

        return LazyMeshFunction(mesh, compute)


__all__ = ["DiffusiveFlux"]
