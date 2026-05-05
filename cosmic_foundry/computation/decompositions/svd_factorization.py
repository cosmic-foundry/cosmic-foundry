"""SVDFactorization: thin-SVD decomposition for minimum-norm pseudoinverse solution."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    ComparisonPredicate,
    DecompositionField,
)
from cosmic_foundry.computation.decompositions.decomposition import DecomposedTensor
from cosmic_foundry.computation.decompositions.factorization import Factorization
from cosmic_foundry.computation.tensor import Tensor, einsum, where


class SVDDecomposedTensor(DecomposedTensor):
    """Thin SVD factors (U, s, Vt) that solve via the Moore-Penrose pseudoinverse.

    Stores the thin SVD A ≈ U Σ Vᵀ and solves A u = b as the minimum-norm
    least-squares solution:

        u† = Vᵀᵀ diag(σᵢ⁻¹) Uᵀ b

    where σᵢ⁻¹ = 1/σᵢ if σᵢ ≥ rcond · σ₀ and 0 otherwise; σ₀ is the
    largest singular value.  This is the Moore-Penrose pseudoinverse
    applied to b.

    For full-rank square systems this recovers the exact solution.  For
    rank-deficient systems it returns the minimum-norm vector in the
    row-space of A — the unique solution with zero component in the null
    space of Aᵀ.

    In plain terms: SVD rotates the problem into a coordinate system where
    A is diagonal (each σᵢ is a scaling factor in one direction).  Solve
    in that system by dividing by each σᵢ, then rotate back.  Very small
    σᵢ signal near-null-space directions; dividing by them amplifies noise,
    so we zero them out below the rcond threshold.

    Use when A may be rank-deficient or ill-conditioned.  Prefer
    LUFactorization for full-rank square systems (same O(N³) cost, smaller
    constant).  Access .u, .s, .vt to inspect condition number or
    null-space structure without a second decomposition.

    Parameters
    ----------
    u:
        Left singular vectors, shape (m, k) where k = min(m, n).
    s:
        Singular values in descending order, shape (k,).
    vt:
        Right singular vectors transposed, shape (k, n).
    rcond:
        Relative threshold: σᵢ < rcond · σ₀ is treated as zero.
    """

    def __init__(
        self,
        u: Tensor,
        s: Tensor,
        vt: Tensor,
        rcond: float,
    ) -> None:
        self._u = u
        self._s = s
        self._vt = vt
        self._rcond = rcond

    @property
    def u(self) -> Tensor:
        """Left singular vectors, shape (m, k)."""
        return self._u

    @property
    def s(self) -> Tensor:
        """Singular values in descending order, shape (k,)."""
        return self._s

    @property
    def vt(self) -> Tensor:
        """Right singular vectors transposed, shape (k, n)."""
        return self._vt

    def solve(self, b: Tensor) -> Tensor:
        """Solve via Moore-Penrose pseudoinverse: u† = Vᵀᵀ diag(σ⁻¹) Uᵀ b."""
        k = self._s.shape[0]
        backend = b.backend
        # threshold = rcond · σ₀  (rank-0 Tensor)
        scale = self._s[0] * Tensor(self._rcond, backend=backend)
        # active[i] — True where σᵢ is above threshold, shape (k,).
        # Uses the same rank-0 / rank-1 comparison pattern as LU's `indices > k`.
        active = self._s >= scale
        # safe_s[i] = σᵢ if active[i] else 1.0 — avoids divide-by-zero.
        # where(rank-1 bool, rank-1, rank-0 scalar): same pattern as LU's
        # `where(indices >= k, col_k.abs(), neg_inf)`.
        safe_s = where(active, self._s, Tensor(1.0, backend=backend))
        # s_inv[i] = 1/σᵢ if active[i] else 0.0
        s_inv = where(active, 1.0 / safe_s, Tensor.zeros(k, backend=backend))
        # Uᵀ b: u is (m, k), b is (m,) → (k,)
        ut_b = einsum("ij,i->j", self._u, b)
        # σ⁻¹ ⊙ (Uᵀ b) → (k,)
        scaled = einsum("i,i->i", s_inv, ut_b)
        # Vᵀᵀ @ scaled: vt is (k, n), scaled is (k,) → (n,)
        return einsum("ij,i->j", self._vt, scaled)


class SVDFactorization(Factorization):
    """Thin-SVD decomposition: A = U Σ Vᵀ.

    Uses Tensor.svd() (thin/economy SVD) to produce an SVDDecomposedTensor
    that solves A u = b via the Moore-Penrose pseudoinverse.

    Cost: O(min(m,n) · max(m,n)²) ≈ O(N³) for square matrices.  Prefer
    LUFactorization for square, full-rank systems (same asymptotic cost,
    smaller constant).  Use SVDFactorization when:
      - A may be rank-deficient (singular stiffness matrices,
        underdetermined systems);
      - a minimum-norm solution is required rather than an exact one;
      - condition-number or null-space analysis is needed alongside solving.

    Parameters
    ----------
    rcond:
        Relative threshold for zeroing out small singular values in the
        pseudoinverse.  Default 1e-10.
    """

    factorization_feasibility_regions = (
        (
            AffineComparisonPredicate(
                {
                    DecompositionField.MATRIX_ROWS: 1.0,
                    DecompositionField.MATRIX_COLUMNS: -1.0,
                },
                "<",
                0.0,
            ),
        ),
        (
            AffineComparisonPredicate(
                {
                    DecompositionField.MATRIX_ROWS: 1.0,
                    DecompositionField.MATRIX_COLUMNS: -1.0,
                },
                ">",
                0.0,
            ),
        ),
        (
            AffineComparisonPredicate(
                {
                    DecompositionField.MATRIX_ROWS: 1.0,
                    DecompositionField.MATRIX_COLUMNS: -1.0,
                },
                "==",
                0.0,
            ),
            ComparisonPredicate(DecompositionField.MATRIX_NULLITY_ESTIMATE, ">", 0),
            ComparisonPredicate(
                DecompositionField.SINGULAR_VALUE_LOWER_BOUND, "<=", 0.0
            ),
        ),
    )

    def __init__(self, rcond: float = 1e-10) -> None:
        self._rcond = rcond

    def factorize(self, a: Tensor) -> SVDDecomposedTensor:
        """Decompose A = U Σ Vᵀ; return an SVDDecomposedTensor."""
        backend = a.backend
        u_raw, s_raw, vt_raw = backend.svd(a._value, a.shape)
        u = Tensor._wrap(u_raw, backend)
        s = Tensor._wrap(s_raw, backend)
        vt = Tensor._wrap(vt_raw, backend)
        return SVDDecomposedTensor(u, s, vt, self._rcond)


__all__ = ["SVDDecomposedTensor", "SVDFactorization"]
