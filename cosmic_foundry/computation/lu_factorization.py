"""LUFactorization: dense LU factorization with partial pivoting."""

from __future__ import annotations

from cosmic_foundry.computation.factorization import FactoredMatrix, Factorization
from cosmic_foundry.computation.tensor import Tensor, einsum


class LUFactoredMatrix(FactoredMatrix):
    """Packed LU factors from which A u = b can be solved by substitution.

    Stores the combined LU matrix (L below diagonal with implicit unit
    diagonal, U on and above), the row-permutation vector, and the set of
    columns whose post-pivot diagonal fell below the singularity threshold.
    Forward and back substitution recover u in O(N²).

    Consistently singular systems (e.g. periodic advection, where the
    matrix has a one-dimensional null space spanned by the constant field)
    are handled transparently: singular columns are pinned to zero
    (minimum-norm, zero-mean convention).  The caller must ensure the RHS
    has zero projection onto the null space; if it does not, the residual
    ‖b − Au‖₂ after the solve will be large.
    """

    def __init__(
        self,
        a_lu: Tensor,
        pivot: list[int],
        singular_cols: set[int],
    ) -> None:
        self._a_lu = a_lu
        self._pivot = pivot
        self._singular_cols = singular_cols

    def solve(self, b: Tensor) -> Tensor:
        """Solve the factored system for rhs b; return u."""
        n = self._a_lu.shape[0]
        a = self._a_lu

        # Forward substitution: Ly = Pb (L has unit diagonal).
        y: Tensor = Tensor([b[self._pivot[i]] for i in range(n)], backend=b.backend)
        for k in range(1, n):
            y[k] = float(y[k]) - float(a[k, :k] @ y[:k])

        # Back substitution: Ux = y.  Null-space components are pinned to 0.
        x: Tensor = Tensor.zeros(n, backend=b.backend)
        for k in range(n - 1, -1, -1):
            if k in self._singular_cols:
                x[k] = 0.0
                continue
            rhs_k = float(y[k])
            if k < n - 1:
                rhs_k -= float(a[k, k + 1 : n] @ x[k + 1 : n])
            x[k] = rhs_k / a[k, k]

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
    """

    def factorize(self, a: Tensor) -> LUFactoredMatrix:
        """Factor A with partial pivoting; return a LUFactoredMatrix."""
        n = a.shape[0]
        a = a.copy()

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

        return LUFactoredMatrix(a, pivot, singular_cols)


__all__ = ["LUFactorization", "LUFactoredMatrix"]
