"""DenseJacobiSolver: Jacobi iteration on the assembled dense stiffness matrix."""

from __future__ import annotations

import math
from typing import cast

from cosmic_foundry.computation.linear_solver import LinearSolver
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction


class DenseJacobiSolver(LinearSolver):
    """Jacobi iterative solver for Lₕ u = f on the assembled N^d × N^d matrix.

    Given the SPD stiffness matrix A assembled via Discretization.assemble_matrix,
    the fixed-point iteration u^{k+1} = D⁻¹(f − (A − D)u^k) is a contraction
    when ρ(I − D⁻¹A) < 1.  For FVMDiscretization(PoissonEquation,
    DiffusiveFlux(order), DirichletBC) on CartesianMesh, this is guaranteed by
    the SPD property proved in C6: SPD implies all eigenvalues of D⁻¹A are
    positive, and the ghost-cell Dirichlet stencil gives A_{ii} > Σ_{j≠i}|A_{ij}|
    for boundary-adjacent rows with A_{ii} = Σ_{j≠i}|A_{ij}| for interior rows
    (weak diagonal dominance everywhere, strict at boundary rows, irreducible mesh
    graph) — by Taussky's theorem D⁻¹A is invertible and Jacobi converges.

    In plain terms: split A = D − (D − A) where D = diag(A).  Each Jacobi
    step solves the trivially-inverted diagonal system for u^{k+1} given u^k.
    Convergence is guaranteed for the Poisson operator at any order; the
    rate is ρ(M_J) = ρ(I − D⁻¹A), which approaches cos(πh) for large N and
    DiffusiveFlux(2) — derived from the Fourier symbol of the tridiagonal
    Laplacian in the limit h → 0.

    All linear algebra is hand-rolled: no NumPy linalg, no LAPACK.  The
    dense matrix assembly scales as O(N^{2d}) in memory and the iteration
    as O(N^{2d}) per step; this solver is intended for small-to-moderate N
    (up to ~32 in 2D) as used in C9 convergence studies.

    Parameters
    ----------
    tol:
        Convergence tolerance on the discrete L²_h residual
        ‖f − A u^k‖_{L²_h} = (Σᵢ |Ωᵢ| (f_i − (Au^k)_i)²)^{1/2}.
    max_iter:
        Maximum number of Jacobi iterations before returning the current
        iterate regardless of residual.
    """

    def __init__(self, tol: float = 1e-10, max_iter: int = 100_000) -> None:
        self._tol = tol
        self._max_iter = max_iter

    def solve(
        self,
        discretization: Discretization,
        rhs: MeshFunction,
    ) -> LazyMeshFunction[float]:
        """Solve Lₕ u = rhs via Jacobi iteration; return the solution MeshFunction."""
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

        # Assemble float matrix from sympy.Matrix (integer entries for rational h)
        a_sym = discretization.assemble_matrix()
        a: list[list[float]] = [
            [float(a_sym[i, j]) for j in range(n)] for i in range(n)
        ]

        # RHS and diagonal vectors
        f: list[float] = [float(rhs(_to_multi(i))) for i in range(n)]  # type: ignore[arg-type]
        diag: list[float] = [a[i][i] for i in range(n)]
        vol: float = float(cast(CartesianMesh, mesh).cell_volume)

        # Jacobi iteration: u^{k+1}_i = (f_i − Σ_{j≠i} A_{ij} u^k_j) / A_{ii}
        u: list[float] = [0.0] * n
        for _ in range(self._max_iter):
            u_new: list[float] = [
                (f[i] - sum(a[i][j] * u[j] for j in range(n) if j != i)) / diag[i]
                for i in range(n)
            ]
            u = u_new
            # Residual ‖f − Au‖_{L²_h}
            residual: float = (
                sum(
                    vol * (f[i] - sum(a[i][j] * u[j] for j in range(n))) ** 2
                    for i in range(n)
                )
            ) ** 0.5
            if residual < self._tol:
                break

        u_list = u

        def _solution(idx: tuple[int, ...]) -> float:
            return u_list[_to_flat(idx)]

        return LazyMeshFunction(mesh, _solution)


__all__ = ["DenseJacobiSolver"]
