"""AdvectiveFlux: NumericalFlux for advection equations F(U) = v·U."""

from __future__ import annotations

from typing import ClassVar, cast

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.advection_operator import AdvectionOperator
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.continuous.manifold import Manifold
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction
from cosmic_foundry.theory.discrete.numerical_flux import NumericalFlux


class AdvectiveFlux(NumericalFlux[sympy.Expr]):
    """Numerical flux for the advective flux F(φ) = v·φ.

    AdvectiveFlux approximates the face-averaged normal flux v·φ(face)·|Aₐ|
    at the interface between two adjacent cells along axis a, using a
    symmetric centered reconstruction from cell averages.

    One class, many instances: AdvectiveFlux(order, continuous_operator) for
    any valid order.  Both satisfy the same Lane C contract at their respective
    orders (see tests/test_convergence_order.py).

    Validity:
        min_order  = 2   — smallest supported order
        order_step = 2   — valid orders are min_order, min_order+step, ...
                           (symmetric stencil design constrains order to even
                           integers: odd error terms vanish by symmetry; only
                           even orders are achievable)

    Stencil derivation (Lane C):

    At order p = 2n, the stencil uses 2n cells (n on each side of the face).
    Coefficients c_0, ..., c_{n-1} are derived symbolically at construction
    by solving the symmetric cell-average even-moment system:

        face_value ≈ Σ_{k=0}^{n-1} c_k · (φ̄_{i+k+1} + φ̄_{i-k})

    where c_k is the coefficient for face offsets ±(2k+1)/2.  The unique
    solution kills all even Taylor error terms through order h^{p-1}, giving
    leading error O(h^p).

    Explicit results: p=2 → (1/2,); p=4 → (7/12, -1/12).  These match the
    standard centered face reconstructions (arithmetic mean for p=2; the
    classical fourth-order interpolant for p=4).

    The assembled stiffness matrix is skew-symmetric (zero diagonal), so
    AdvectiveFlux is incompatible with DenseJacobiSolver.  It contributes
    only _OrderClaim entries to the convergence test suite.

    Parameters
    ----------
    order:
        Integer satisfying order >= min_order and
        (order - min_order) % order_step == 0.  Stencil coefficients are
        derived via SymPy at construction time; __call__ uses only arithmetic.
    manifold:
        The manifold on which the operator acts; used to construct the
        AdvectionOperator (v·φ: Ω⁰ → Ω¹) that this instance approximates.

    __call__ signature:
        (U: MeshFunction) -> MeshFunction

        U        — cell averages (MeshFunction callable with cell index)
        returns  — face-flux MeshFunction callable as result((axis, idx_low))
                   where idx_low is the low-side cell index of the face;
                   the high cell is idx_low with idx_low[axis] + 1.
    """

    min_order: ClassVar[int] = 2
    order_step: ClassVar[int] = 2

    def __init__(self, order: int, manifold: Manifold) -> None:
        if order < self.min_order or (order - self.min_order) % self.order_step != 0:
            raise ValueError(
                f"AdvectiveFlux order must be >= {self.min_order} and satisfy "
                f"(order - {self.min_order}) % {self.order_step} == 0; got {order}"
            )
        self._order = order
        self._continuous_operator = AdvectionOperator(manifold)

        # Derive stencil coefficients from first principles.
        #
        # Goal: find c_0, ..., c_{n-1} (n = order//2) such that the
        # symmetric stencil
        #
        #   face_value ≈ Σ_{k=0}^{n-1} c_k (φ̄_{+ξ_k} + φ̄_{-ξ_k})
        #
        # approximates φ(face) to O(hᵖ), where ξ_k = (2k+1)/2 are the
        # positive cell-center offsets from the face in units of h.
        #
        # Cell-average Taylor expansion (in units of h):
        #
        #   φ̄_ξ = ∫_{ξ-1/2}^{ξ+1/2} φ(face + sh) ds
        #        = Σ_{m≥0} (φ^(m)(face) / m!) h^m · M_m(ξ)
        #
        # The symmetric sum φ̄_{+ξ} + φ̄_{-ξ} retains only even m
        # (M_m(−ξ) = (−1)^m M_m(ξ), so odd m cancels).  The stencil sum becomes
        #
        #   Σ_k c_k (φ̄_{+ξ_k} + φ̄_{-ξ_k})
        #   = Σ_{m even} (φ^(m)(face) / m!) h^m · 2 Σ_k c_k M_m(ξ_k)
        #
        # Matching to φ(face) requires:
        #   m=0: 2 Σ_k c_k M_0(ξ_k) = 1        (pin the face value)
        #   m=2,4,...,2n-2: 2 Σ_k c_k M_m(ξ_k) = 0  (kill error through h^{p-1})
        #
        # These n conditions in n unknowns have a unique solution; solving them
        # yields O(hᵖ) accuracy.  (Odd m vanish by symmetry; m ≥ 2n give
        # error O(hᵖ) after multiplying by h^m.)

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
            for m in range(0, 2 * n, 2)
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
            face_area: sympy.Expr = mesh.face_area(axis)

            def shift(idx: tuple[int, ...], delta: int) -> tuple[int, ...]:
                return idx[:axis] + (idx[axis] + delta,) + idx[axis + 1 :]

            face_value = sum(
                c_k * (U(shift(idx_low, k + 1)) + U(shift(idx_low, -k)))  # type: ignore[arg-type]
                for k, c_k in enumerate(self._coeffs)
            )
            return face_value * face_area

        return LazyMeshFunction(mesh, compute)


__all__ = ["AdvectiveFlux"]
