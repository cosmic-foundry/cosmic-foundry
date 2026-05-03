"""LUFactorization: dense LU factorization with partial pivoting."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.algorithm_capabilities import (
    CONDITION_LIMIT,
    ComparisonPredicate,
    DecompositionField,
)
from cosmic_foundry.computation.decompositions.decomposition import DecomposedTensor
from cosmic_foundry.computation.decompositions.factorization import Factorization
from cosmic_foundry.computation.tensor import Tensor, arange, einsum, where

# Diagonal magnitude below which a pivot is treated as singular.
_SINGULAR_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# Module-level loop bodies — stable Python objects so JAX can cache the
# traced XLA computation across calls.  All per-call constants (indices,
# neg_inf, zeros_n) travel through the state tuple rather than being closed
# over, keeping the function identity fixed regardless of matrix size.
# ---------------------------------------------------------------------------


def _factorize_body(k: Any, state: tuple) -> tuple:
    a, pivot, is_singular, indices, neg_inf, zeros_n = state

    # Find pivot: argmax of |col_k| for rows >= k.
    col_k = a[(slice(None), k)]
    masked_col = where(indices >= k, tensor.abs(col_k), neg_inf)
    max_row = tensor.argmax(masked_col)

    # Swap rows k and max_row.
    row_k = tensor.copy(a[k])
    a = a.set(k, a[max_row])
    a = a.set(max_row, row_k)

    # Swap pivot entries.
    piv_k = pivot[k]
    piv_mr = pivot[max_row]
    pivot = pivot.set(k, piv_mr)
    pivot = pivot.set(max_row, piv_k)

    # Singularity check; replace near-zero diagonal with 1 to avoid
    # divide-by-zero in the factor computation (singular flag records it).
    row_k_new = a[k]
    diag_k = row_k_new[k]
    sing_k = tensor.abs(diag_k) < _SINGULAR_TOL
    is_singular = is_singular.set(k, sing_k)
    safe_diag = where(sing_k, Tensor(1.0, backend=a.backend), diag_k)
    a = a.set((k, k), safe_diag)

    # Multipliers for rows > k (zero elsewhere so shapes stay fixed).
    col_k_new = a[(slice(None), k)]
    factors = where(indices > k, col_k_new / safe_diag, zeros_n)

    # Rank-1 update: zero out columns <= k to avoid corrupting stored
    # L factors from prior steps; column k is also zeroed (mask is
    # indices > k, so col k maps to 0) leaving a[j,k] unchanged for
    # the subsequent store step.
    pivot_row = a[k]
    pivot_row_masked = where(indices > k, pivot_row, zeros_n)
    a = a - einsum("i,j->ij", factors, pivot_row_masked)

    # Store multipliers in the lower triangle of column k.  After the
    # rank-1 update, a[j,k] for j > k still holds the original value
    # (the update zeroed column k via the mask); overwrite with factors.
    col_k_post = a[(slice(None), k)]
    a = a.set((slice(None), k), where(indices > k, factors, col_k_post))

    return (a, pivot, is_singular, indices, neg_inf, zeros_n)


def _fwd_body(k: Any, state: tuple) -> tuple:
    y, a_lu, indices, zeros_n = state
    row_k = a_lu[k]
    masked_row = where(indices < k, row_k, zeros_n)
    corr = masked_row @ y
    y = y.set(k, y[k] - corr)
    return (y, a_lu, indices, zeros_n)


def _back_body(k_fwd: Any, state: tuple) -> tuple:
    # n_minus_1 is a 0-d int Tensor so k stays a Tensor for all downstream ops.
    x, y, a_lu, is_singular, indices, zeros_n, n_minus_1 = state
    k = n_minus_1 - k_fwd
    row_k = a_lu[k]
    masked_row = where(indices > k, row_k, zeros_n)
    corr = masked_row @ x
    diag_k = row_k[k]
    sing_k = is_singular[k]
    rhs = y[k] - corr
    new_xk = where(sing_k, Tensor(0.0, backend=x.backend), rhs / diag_k)
    x = x.set(k, new_xk)
    return (x, y, a_lu, is_singular, indices, zeros_n, n_minus_1)


class LUDecomposedTensor(DecomposedTensor):
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
    ‖b − Au‖₂ after the solve will be large.  For rank-deficient systems
    where the null-space structure is unknown, prefer SVDFactorization —
    it identifies and handles rank deficiency automatically.

    Both substitution passes use backend.fori_loop with module-level body
    functions so JAX can cache the traced XLA computation across solves.
    Each loop body operates on full-N vectors with masks rather than
    varying-length slices, keeping every shape static across iterations.
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
        backend = self._a_lu.backend
        indices = arange(n, backend=backend)
        zeros_n = Tensor.zeros(n, backend=backend)
        n_minus_1: Tensor = Tensor(n - 1, backend=backend)

        # Forward substitution: Ly = Pb  (L has unit diagonal).
        y: Tensor = tensor.take(b, self._pivot)
        y, *_ = backend.fori_loop(n, _fwd_body, (y, self._a_lu, indices, zeros_n))

        # Back substitution: Ux = y.
        x: Tensor = Tensor.zeros(n, backend=backend)
        x, *_ = backend.fori_loop(
            n,
            _back_body,
            (x, y, self._a_lu, self._is_singular, indices, zeros_n, n_minus_1),
        )
        return x


class LUFactorization(Factorization):
    """Dense LU factorization with partial pivoting: PA = LU.

    Gaussian elimination with partial pivoting places the largest-magnitude
    entry in each pivot column on the diagonal, making the factorization
    numerically stable for any non-singular A, including indefinite and
    skew-symmetric matrices where Jacobi iteration diverges.

    The factorization is performed on a copy of A; the original matrix is
    not modified.  All loop bodies use backend.fori_loop so the entire
    factorization compiles to a single XLA kernel on JaxBackend.

    Each iteration of the elimination loop operates on full-N vectors and
    an N×N outer-product update rather than varying-length submatrix
    slices.  Columns and rows outside the active (k+1:n, k+1:n) block are
    zeroed by masks, keeping every shape static so JAX traces the body
    once regardless of N.  Memory and time scale as O(N²) and O(N³)
    respectively.

    The loop body is a module-level function (_factorize_body) so JAX can
    cache the traced XLA computation across calls with the same matrix size.
    The pivot search uses a masked argmax over each column rather than a
    Python data-dependent branch, so the factorize loop body is compatible
    with JAX tracing (no Python bool over traced values).
    """

    factorization_feasibility_certificate = (
        ComparisonPredicate(DecompositionField.SINGULAR_VALUE_LOWER_BOUND, ">", 0.0),
        ComparisonPredicate(
            DecompositionField.CONDITION_ESTIMATE, "<=", CONDITION_LIMIT
        ),
    )

    def factorize(self, a: Tensor) -> LUDecomposedTensor:
        """Factor A with partial pivoting; return a LUDecomposedTensor."""
        n = a.shape[0]
        backend = a.backend
        a = tensor.copy(a)

        pivot: Tensor = arange(n, backend=backend)
        is_singular: Tensor = Tensor([False] * n, backend=backend)
        indices: Tensor = arange(n, backend=backend)
        neg_inf: Tensor = Tensor(-1e300, backend=backend)
        zeros_n: Tensor = Tensor.zeros(n, backend=backend)

        a_f, pivot_f, is_singular_f, *_ = backend.fori_loop(
            n, _factorize_body, (a, pivot, is_singular, indices, neg_inf, zeros_n)
        )
        return LUDecomposedTensor(a_f, pivot_f, is_singular_f)


__all__ = ["LUDecomposedTensor", "LUFactorization"]
