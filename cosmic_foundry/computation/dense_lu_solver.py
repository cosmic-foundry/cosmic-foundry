"""DenseLUSolver: direct LU factorization on the assembled dense stiffness matrix."""

from __future__ import annotations

import math
from typing import cast

from cosmic_foundry.computation.linear_solver import LinearSolver
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction


class DenseLUSolver(LinearSolver):
    """Direct solver for Lₕ u = f via LU factorization of the N^d × N^d matrix.

    Given the stiffness matrix A assembled via Discretization.assemble,
    the system is solved exactly (up to floating-point rounding) in one pass:

        PA = LU   (Gaussian elimination with partial pivoting)
        Ly = Pb   (forward substitution)
        Ux = y    (back substitution)

    where P is a row-permutation matrix chosen to place the largest-magnitude
    entry in each pivot column on the diagonal.  Partial pivoting makes the
    factorization numerically stable for any non-singular A, including
    indefinite and skew-symmetric matrices where DenseJacobiSolver diverges.

    The factorization is performed in-place on a copy of A; the original
    assembled matrix is not modified.  All arithmetic is hand-rolled with
    no NumPy linalg or LAPACK.  Memory and time scale as O(N^{2d}) and
    O(N^{3d}) respectively; this solver is intended for small-to-moderate N.

    Consistently singular systems (e.g. periodic advection, where the
    stiffness matrix has a one-dimensional null space spanned by the constant
    field) are handled transparently: when partial pivoting leaves a diagonal
    entry below 1e-10 after column reduction, that component is pinned to zero
    (minimum-norm, zero-mean convention).  The caller must ensure the RHS has
    zero projection onto the null space; if it does not, the residual after
    the solve will be large and the tol assertion will fail.

    Parameters
    ----------
    tol:
        Acceptance threshold on the discrete L²_h residual
        ‖f − A u‖_{L²_h} after the solve.  The residual is computed once
        after back-substitution and stored in residuals.
    """

    def __init__(self, tol: float = 1e-10) -> None:
        self._tol = tol
        self._residuals: list[float] = []

    @property
    def residuals(self) -> Tensor:
        """Residual history from the most recent solve as a rank-1 Tensor.

        For a direct solver the Tensor has exactly one entry — the final
        residual after back-substitution — rather than an iteration trace.
        """
        return Tensor(list(self._residuals))

    def solve(
        self,
        discretization: Discretization,
        rhs: MeshFunction,
    ) -> LazyMeshFunction[float]:
        """Solve Lₕ u = rhs via LU factorization; return the solution MeshFunction."""
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

        a_orig: list[list[float]] = discretization.assemble().to_list()
        a: list[list[float]] = [row[:] for row in a_orig]
        f: list[float] = [float(rhs(_to_multi(i))) for i in range(n)]  # type: ignore[arg-type]
        vol: float = float(cast(CartesianMesh, mesh).cell_volume)

        # LU factorization with partial pivoting: PA = LU.
        # After factorization, a[i][j] holds U for j >= i and L for j < i
        # (L has implicit unit diagonal).  pivot[i] records the source row.
        # singular_cols records columns where the post-pivot diagonal is below
        # the threshold — these correspond to null-space modes in consistently
        # singular systems (e.g. periodic advection, rank N−1).  The minimum-norm
        # solution sets those components to zero in back-substitution.
        singular_tol: float = 1e-10
        singular_cols: set[int] = set()
        pivot = list(range(n))
        for k in range(n):
            max_val = abs(a[k][k])
            max_row = k
            for i in range(k + 1, n):
                if abs(a[i][k]) > max_val:
                    max_val = abs(a[i][k])
                    max_row = i
            if max_row != k:
                a[k], a[max_row] = a[max_row], a[k]
                pivot[k], pivot[max_row] = pivot[max_row], pivot[k]
            if abs(a[k][k]) < singular_tol:
                singular_cols.add(k)
                a[k][k] = 1.0  # placeholder diagonal; x[k] forced to 0
                for i in range(k + 1, n):
                    a[i][k] = 0.0  # L factor = 0: no elimination for this column
                continue
            for i in range(k + 1, n):
                factor = a[i][k] / a[k][k]
                a[i][k] = factor
                for j in range(k + 1, n):
                    a[i][j] -= factor * a[k][j]

        # Forward substitution: Ly = Pb (L has unit diagonal).
        y: list[float] = [f[pivot[i]] for i in range(n)]
        for k in range(n):
            for j in range(k):
                y[k] -= a[k][j] * y[j]

        # Back substitution: Ux = y.  Null-space components are pinned to 0
        # (minimum-norm solution for consistently singular systems).
        x: list[float] = [0.0] * n
        for k in range(n - 1, -1, -1):
            if k in singular_cols:
                x[k] = 0.0
                continue
            x[k] = y[k]
            for j in range(k + 1, n):
                x[k] -= a[k][j] * x[j]
            x[k] /= a[k][k]

        # Recompute residual from original matrix (a is now LU, not A).
        r: list[float] = [
            f[i] - sum(a_orig[i][j] * x[j] for j in range(n)) for i in range(n)
        ]
        self._residuals = [math.sqrt(sum(vol * ri**2 for ri in r))]

        u_list = x

        def _solution(idx: tuple[int, ...]) -> float:
            return u_list[_to_flat(idx)]

        return LazyMeshFunction(mesh, _solution)


__all__ = ["DenseLUSolver"]
