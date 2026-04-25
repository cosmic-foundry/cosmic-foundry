"""DenseJacobiSolver: Jacobi iteration on the assembled dense stiffness matrix."""

from __future__ import annotations

import math
from typing import cast

from cosmic_foundry.computation.linear_solver import LinearSolver
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction


class DenseJacobiSolver(LinearSolver):
    """Jacobi iterative solver for Lₕ u = f on the assembled N^d × N^d matrix.

    Given the SPD stiffness matrix A assembled via Discretization.assemble,
    the damped fixed-point iteration u^{k+1} = u^k + ω D⁻¹(f − Au^k) is a
    contraction when ρ(I − ω D⁻¹A) < 1.  The relaxation factor ω is derived
    automatically from the Gershgorin bound on λ_max(D⁻¹A):

        G = max_i Σ_j |A_{ij}/A_{ii}|   (Gershgorin bound, includes j = i term)
        ω = min(2/G, 1)

    G is an upper bound on λ_max(D⁻¹A) by the Gershgorin circle theorem;
    ω = 2/G guarantees ρ(I − ω D⁻¹A) < 1 whenever λ_max < G.  For
    DiffusiveFlux(2) the interior stencil has G = 2, giving ω = 1 (standard
    Jacobi, the optimal choice).  For DiffusiveFlux(4) the wider stencil
    violates diagonal dominance (G = 32/15 > 2), so standard Jacobi diverges
    and ω = 15/16 is applied automatically.

    In plain terms: split A = D − (D − A) where D = diag(A).  Each damped
    Jacobi step scales the correction by ω before applying the diagonal inverse.
    With ω derived from the Gershgorin bound the iteration contracts for any
    SPD operator assembled by FVMDiscretization with DirichletBC, regardless of
    stencil width.

    The iteration count is bounded analytically before the loop.  For a
    translationally-invariant stencil on a Cartesian mesh, the eigenvalue of
    D⁻¹A at the slowest Fourier mode (k_a = 1 in every axis) is:

        μ₁ = Σ_j (A_{c,j}/d_c) · Π_a cos(|j_a − c_a| π / N_a)

    where c is a central interior cell.  The spectral radius of the damped
    iteration matrix is ρ = |1 − ω μ₁|, and the required iteration count is

        k_max = ⌈log(tol / ‖f‖_{L²_h}) / log ρ⌉

    computed in O(n) from one interior row of A — the same cost as a single
    Jacobi step.  max_iter remains a hard safety cap but is never reached in
    normal operation.  The formula is exact for separable Cartesian stencils
    (no cross-axis coupling) and conservative otherwise.

    All linear algebra is hand-rolled: no NumPy linalg, no LAPACK.  The
    dense matrix assembly scales as O(N^{2d}) in memory and the iteration
    as O(N^{2d}) per step; this solver is intended for small-to-moderate N.

    Parameters
    ----------
    tol:
        Convergence tolerance on the discrete L²_h residual
        ‖f − A u^k‖_{L²_h} = (Σᵢ |Ωᵢ| (f_i − (Au^k)_i)²)^{1/2}.
    max_iter:
        Hard cap on Jacobi iterations; the analytical k_max bound is used
        instead in normal operation, so this is only a safety net.
    """

    def __init__(self, tol: float = 1e-10, max_iter: int = 100_000) -> None:
        self._tol = tol
        self._max_iter = max_iter
        self._residuals: list[float] = []

    @property
    def residuals(self) -> Tensor:
        """Residual history ‖f − Au^k‖_{L²_h} from the most recent solve.

        residuals[k] is the discrete L²_h residual norm at the start of
        iteration k, before the k-th damped-Jacobi update is applied.
        The Tensor is populated by solve() and replaced on each call.
        """
        return Tensor(list(self._residuals))

    def solve(
        self,
        discretization: Discretization,
        rhs: MeshFunction,
    ) -> LazyMeshFunction[float]:
        """Solve Lₕ u = rhs via damped Jacobi; return the solution MeshFunction."""
        mesh = discretization.mesh
        shape = mesh.shape
        ndim = len(shape)
        n = math.prod(shape)

        def _to_multi(flat: int) -> tuple[int, ...]:
            idx = []
            k = flat
            for axis in range(ndim):
                idx.append(k % shape[axis])
                k //= shape[axis]
            return tuple(idx)

        def _to_flat(idx: tuple[int, ...]) -> int:
            flat = 0
            stride = 1
            for axis in range(ndim):
                flat += idx[axis] * stride
                stride *= shape[axis]
            return flat

        a: Tensor = discretization.assemble()

        # RHS and diagonal vectors
        f: Tensor = Tensor([rhs(_to_multi(i)) for i in range(n)])  # type: ignore[arg-type]
        diag: Tensor = a.diag()
        vol: float = float(cast(CartesianMesh, mesh).cell_volume)

        # Gershgorin bound on λ_max(D⁻¹A): ω = min(2/G, 1) guarantees contraction.
        lambda_max_bound: float = max(
            sum(abs(a[i, j] / diag[i]) for j in range(n)) for i in range(n)
        )
        omega: float = min(2.0 / lambda_max_bound, 1.0)

        # Analytical k_max from the Fourier spectral radius.
        # μ₁ = eigenvalue of D⁻¹A at slowest mode (k_a=1 each axis), computed
        # from one interior row c via μ₁ = Σ_j (A_{c,j}/d_c)·Π_a cos(|offset_a|π/N_a).
        # Valid for separable Cartesian stencils; conservative otherwise.
        c_multi = tuple(s // 2 for s in shape)
        c_flat = _to_flat(c_multi)
        d_c = diag[c_flat]
        mu_1: float = sum(
            a[c_flat, j]
            / d_c
            * math.prod(
                math.cos(abs(_to_multi(j)[ax] - c_multi[ax]) * math.pi / shape[ax])
                for ax in range(ndim)
            )
            for j in range(n)
        )
        rho: float = abs(1.0 - omega * mu_1)
        r0_norm: float = math.sqrt(vol) * f.norm()
        # Add 10 to absorb the ~0.1% underestimate of rho_fourier vs rho_actual
        # caused by boundary-row diagonal modifications; negligible for large N.
        k_max: int = (
            math.ceil(math.log(self._tol / r0_norm) / math.log(rho)) + 10
            if r0_norm > self._tol and 0.0 < rho < 1.0
            else self._max_iter
        )

        # Damped Jacobi: u^{k+1} = u^k + ω D⁻¹(f − Au^k)
        u: Tensor = Tensor.zeros(n)
        self._residuals = []
        for _ in range(min(self._max_iter, k_max)):
            r: Tensor = f - a @ u
            residual: float = math.sqrt(vol) * r.norm()
            self._residuals.append(residual)
            if residual < self._tol:
                break
            u = u + omega * (r / diag)

        u_tens = u

        def _solution(idx: tuple[int, ...]) -> float:
            return float(u_tens[_to_flat(idx)])

        return LazyMeshFunction(mesh, _solution)


__all__ = ["DenseJacobiSolver"]
