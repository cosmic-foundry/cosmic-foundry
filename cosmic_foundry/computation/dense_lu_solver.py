"""DenseLUSolver: direct LU factorization on a dense matrix."""

from __future__ import annotations

from cosmic_foundry.computation.linear_solver import LinearSolver
from cosmic_foundry.computation.tensor import Tensor, einsum


class DenseLUSolver(LinearSolver):
    """Direct solver for A u = b via LU factorization of the N × N matrix.

    The system is solved exactly (up to floating-point rounding) in one pass:

        PA = LU   (Gaussian elimination with partial pivoting)
        Ly = Pb   (forward substitution)
        Ux = y    (back substitution)

    where P is a row-permutation matrix chosen to place the largest-magnitude
    entry in each pivot column on the diagonal.  Partial pivoting makes the
    factorization numerically stable for any non-singular A, including
    indefinite and skew-symmetric matrices where DenseJacobiSolver diverges.

    The factorization is performed in-place on a copy of A; the original
    matrix is not modified.  The elimination step is vectorized: each
    outer-product rank-1 update uses Tensor slice operations so that
    NumpyBackend delegates to BLAS.  Forward and back substitution are
    sequential by necessity but use Tensor dot products for each row.
    Memory and time scale as O(N²) and O(N³) respectively; this solver is
    intended for small-to-moderate N.

    Consistently singular systems (e.g. periodic advection, where the matrix
    has a one-dimensional null space spanned by the constant field) are handled
    transparently: when partial pivoting leaves a diagonal entry below 1e-10
    after column reduction, that component is pinned to zero (minimum-norm,
    zero-mean convention).  The caller must ensure the RHS has zero projection
    onto the null space; if it does not, the residual ‖b − Au‖₂ after the
    solve will be large.
    """

    def __init__(self) -> None:
        pass

    def solve(self, a: Tensor, b: Tensor) -> Tensor:
        """Solve A u = b via LU factorization; return the solution Tensor."""
        n = a.shape[0]

        a_orig: Tensor = a
        a = a_orig.copy()

        # LU factorization with partial pivoting: PA = LU.
        # a[i,j] holds U for j >= i and L for j < i (L has implicit unit diagonal).
        # singular_cols records columns where the post-pivot diagonal is below the
        # threshold — these correspond to null-space modes in consistently singular
        # systems.  The minimum-norm solution sets those components to zero.
        singular_tol: float = 1e-10
        singular_cols: set[int] = set()
        pivot = list(range(n))

        for k in range(n):
            # Pivot search: O(N) Python loop, cheap relative to the O(N²) update.
            max_val = abs(a[k, k])
            max_row = k
            for i in range(k + 1, n):
                v = a[i, k]
                if abs(v) > max_val:
                    max_val = abs(v)
                    max_row = i

            if max_row != k:
                # Deep-copy row k before overwriting so NumpyBackend (in-place writes)
                # does not corrupt the saved value via aliased views.
                row_k = a[k].copy()
                a[k] = a[max_row]
                a[max_row] = row_k
                pivot[k], pivot[max_row] = pivot[max_row], pivot[k]

            if abs(a[k, k]) < singular_tol:
                singular_cols.add(k)
                a[k, k] = 1.0
                continue

            if k + 1 < n:
                # Vectorized rank-1 update: eliminates column k below the pivot.
                # factor_col = a[k+1:n, k] / a[k, k]  (L factor column)
                # a[k+1:n, k+1:n] -= outer(factor_col, a[k, k+1:n])
                factor_col: Tensor = a[k + 1 : n, k] / a[k, k]
                a[k + 1 : n, k] = factor_col
                row_pivot: Tensor = a[k, k + 1 : n]
                a[k + 1 : n, k + 1 : n] = a[k + 1 : n, k + 1 : n] - einsum(
                    "i,j->ij", factor_col, row_pivot
                )

        # Forward substitution: Ly = Pb (L has unit diagonal).
        y: Tensor = Tensor([b[pivot[i]] for i in range(n)], backend=b.backend)
        for k in range(1, n):
            y[k] = float(y[k]) - float(a[k, :k] @ y[:k])

        # Back substitution: Ux = y.  Null-space components are pinned to 0.
        x: Tensor = Tensor.zeros(n, backend=b.backend)
        for k in range(n - 1, -1, -1):
            if k in singular_cols:
                x[k] = 0.0
                continue
            rhs_k = float(y[k])
            if k < n - 1:
                rhs_k -= float(a[k, k + 1 : n] @ x[k + 1 : n])
            x[k] = rhs_k / a[k, k]

        return x


__all__ = ["DenseLUSolver"]
