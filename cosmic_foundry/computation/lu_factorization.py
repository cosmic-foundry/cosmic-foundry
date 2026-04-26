"""LUFactorization: dense LU factorization with partial pivoting."""

from __future__ import annotations

from cosmic_foundry.computation.factorization import FactoredMatrix, Factorization
from cosmic_foundry.computation.tensor import Tensor, arange, einsum, where


class LUFactoredMatrix(FactoredMatrix):
    """Packed LU factors from which A u = b can be solved by substitution.

    Stores the combined LU matrix (L below diagonal with implicit unit
    diagonal, U on and above), the row-permutation vector as an integer
    Tensor, and a bool Tensor marking columns whose post-pivot diagonal
    fell below the singularity threshold.

    Forward and back substitution recover u in O(N²).  Consistently
    singular systems (e.g. periodic advection, where the matrix has a
    one-dimensional null space spanned by the constant field) are handled
    transparently: singular columns are pinned to zero (minimum-norm,
    zero-mean convention).  The caller must ensure the RHS has zero
    projection onto the null space; if it does not, the residual
    ‖b − Au‖₂ after the solve will be large.
    """

    def __init__(
        self,
        a_lu: Tensor,
        pivot: Tensor,
        is_singular: Tensor,
    ) -> None:
        self._a_lu = a_lu
        self._pivot = pivot
        self._is_singular = is_singular

    def solve(self, b: Tensor) -> Tensor:
        """Solve the factored system for rhs b; return u."""
        n = self._a_lu.shape[0]
        a = self._a_lu

        # Forward substitution: Ly = Pb  (L has unit diagonal).
        # b.take(pivot) gathers b reordered by the row permutation.
        y: Tensor = b.take(self._pivot)
        for k in range(1, n):
            y[k] = y.element(k) - a[k, :k] @ y[:k]

        # Back substitution: Ux = y.
        # Singular columns are pinned to zero (minimum-norm convention).
        x: Tensor = Tensor.zeros(n, backend=b.backend)
        for k in range(n - 1, -1, -1):
            rhs_k: Tensor = y.element(k)
            if k < n - 1:
                rhs_k = rhs_k - a[k, k + 1 : n] @ x[k + 1 : n]
            diag_k: Tensor = a.element(k, k)
            sing_k: Tensor = self._is_singular.element(k)
            x[k] = where(sing_k, 0.0, rhs_k / diag_k)

        return x


class LUFactorization(Factorization):
    """Dense LU factorization with partial pivoting: PA = LU.

    Gaussian elimination with partial pivoting places the largest-magnitude
    entry in each pivot column on the diagonal, making the factorization
    numerically stable for any non-singular A, including indefinite and
    skew-symmetric matrices where Jacobi iteration diverges.

    The factorization is performed in-place on a copy of A; the original
    matrix is not modified.  The elimination step is vectorized: each
    outer-product rank-1 update uses Tensor slice operations so that
    NumpyBackend delegates to BLAS.  Forward and back substitution are
    sequential by necessity but use Tensor dot products for each row.
    Memory and time scale as O(N²) and O(N³) respectively; this solver is
    intended for small-to-moderate N.

    The pivot search uses a masked argmax over each column rather than a
    Python data-dependent branch, so the factorize loop body is compatible
    with JAX tracing (no Python bool over traced values).
    """

    _SINGULAR_TOL: float = 1e-10

    def factorize(self, a: Tensor) -> LUFactoredMatrix:
        """Factor A with partial pivoting; return a LUFactoredMatrix."""
        n = a.shape[0]
        a = a.copy()

        # a[i,j] holds U for j >= i and L for j < i (L has implicit unit diagonal).
        pivot: Tensor = arange(n, backend=a.backend)
        is_singular: Tensor = Tensor([False] * n, backend=a.backend)

        # Precompute a row-index tensor for masked argmax pivot search.
        indices: Tensor = arange(n, backend=a.backend)
        neg_inf: Tensor = Tensor(-1e300, backend=a.backend)

        for k in range(n):
            # Pivot search: mask entries above row k with -∞, find argmax of |col k|.
            col_k: Tensor = a[:, k]
            masked: Tensor = where(indices >= k, col_k.abs(), neg_inf)
            max_row = masked.argmax()

            # Unconditional row swap (identity when max_row == k).
            row_k_copy: Tensor = a[k].copy()
            a[k] = a[max_row]
            a[max_row] = row_k_copy
            old_pivot_k = pivot[k]
            old_pivot_mr = pivot[max_row]
            pivot[k] = old_pivot_mr
            pivot[max_row] = old_pivot_k

            # Singular detection: stabilize the diagonal to avoid division by zero.
            diag_k: Tensor = a.element(k, k)
            sing_k: Tensor = diag_k.abs() < self._SINGULAR_TOL
            is_singular[k] = sing_k
            safe_diag: Tensor = where(sing_k, 1.0, diag_k)
            a[k, k] = safe_diag

            if k + 1 < n:
                # Vectorized rank-1 update: eliminates column k below the pivot.
                # factor_col = a[k+1:n, k] / safe_diag  (L factor column)
                # If singular, zero out factor_col so the update is a no-op.
                zero_col: Tensor = Tensor.zeros(n - k - 1, backend=a.backend)
                factor_col: Tensor = where(
                    sing_k, zero_col, a[k + 1 : n, k] / safe_diag
                )
                a[k + 1 : n, k] = factor_col
                row_pivot: Tensor = a[k, k + 1 : n]
                a[k + 1 : n, k + 1 : n] = a[k + 1 : n, k + 1 : n] - einsum(
                    "i,j->ij", factor_col, row_pivot
                )

        return LUFactoredMatrix(a, pivot, is_singular)


__all__ = ["LUFactorization", "LUFactoredMatrix"]
