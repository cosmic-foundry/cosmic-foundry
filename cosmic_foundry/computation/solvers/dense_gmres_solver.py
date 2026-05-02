"""DenseGMRESSolver: matrix-free GMRES(k) iteration."""

from __future__ import annotations

from typing import Any, NamedTuple

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.algorithm_capabilities import (
    ComparisonPredicate,
    MembershipPredicate,
)
from cosmic_foundry.computation.decompositions.svd_factorization import SVDFactorization
from cosmic_foundry.computation.solvers._capability_claims import (
    CONDITION_LIMIT,
    LINEARITY_TOLERANCE,
    LinearSolverCapability,
    Provision,
    Requirement,
    budget_predicates,
    capability,
    contract,
    linear_system_predicates,
)
from cosmic_foundry.computation.solvers.iterative_solver import KrylovSolver
from cosmic_foundry.computation.solvers.linear_solver import LinearOperator
from cosmic_foundry.computation.tensor import Tensor


class _GMRESState(NamedTuple):
    u: Tensor  # current solution, shape (N,)
    r: Tensor  # residual b − op.apply(u), shape (N,)
    b: Tensor  # RHS, shape (N,)
    iteration: Tensor  # restart cycle count, rank-0 int Tensor


class DenseGMRESSolver(KrylovSolver):
    """GMRES(k) solver for A u = b; A must be non-singular (not necessarily SPD).

    Each restart cycle builds a rank-k Krylov subspace

        K_k = span{r₀, Ar₀, …, A^{k−1}r₀}

    via the Arnoldi process and finds the y ∈ Rᵏ that minimizes ‖β e₁ − H̃ y‖₂,
    where H̃ ∈ R^{(k+1)×k} is the upper Hessenberg matrix and β = ‖r₀‖₂.  The
    solution update is u ← u + Vₖ y where the columns of Vₖ are the k Arnoldi
    basis vectors.

    Unlike CG (which requires A to be SPD), GMRES minimizes the residual over
    the full Krylov subspace without assuming symmetry or positive definiteness.
    CG takes one step per Krylov vector and is cheaper per iteration; GMRES(k)
    stores all k basis vectors and solves a small least-squares problem at the
    end of each cycle.

    Restart structure: after k Arnoldi steps the basis is discarded and the
    process restarts from the current residual.  Restart limits memory to O(k N)
    and arithmetic to O(k² N) per cycle.

    Orthogonalization uses modified Gram-Schmidt via a masked full inner product:
    h_all = V @ w computes all k+1 inner products; indices beyond the current
    step j are zeroed so that the fori_loop carry has fixed shape, enabling
    compilation to a single XLA kernel on JaxBackend.

    The Hessenberg least-squares problem min ‖β e₁ − H̃ y‖₂ is solved via the
    Moore-Penrose pseudoinverse (SVDFactorization) so that breakdown (h_{j+1,j}≈0,
    i.e. Krylov subspace exhausted) is handled gracefully rather than causing a
    divide-by-zero.

    Prefer DenseCGSolver for SPD systems.  Use DenseGMRESSolver for non-symmetric
    or indefinite matrices (e.g. advection-dominated flows, non-self-adjoint
    operators) where CG does not apply.

    Parameters
    ----------
    tol:
        Convergence tolerance: iteration stops when ‖r‖₂² < tol².
    max_iter:
        Hard cap on restart cycles.
    restart:
        Arnoldi basis size k per restart cycle.
    """

    def __init__(
        self, tol: float = 1e-10, max_iter: int = 1000, restart: int = 20
    ) -> None:
        self._tol = tol
        self._max_iter = max_iter
        self._restart = restart

    def init_state(self, op: LinearOperator, b: Tensor) -> _GMRESState:
        u = Tensor.zeros(b.shape[0], backend=b.backend)
        r = tensor.copy(b)  # r₀ = b − A·0 = b
        iteration: Tensor = Tensor(0, backend=b.backend)
        return _GMRESState(u, r, b, iteration)

    def step(self, op: LinearOperator, state: Any) -> _GMRESState:
        s: _GMRESState = state
        k = self._restart
        n = s.b.shape[0]
        backend = s.b.backend

        beta = tensor.norm(s.r)
        V = Tensor.zeros(k + 1, n, backend=backend)
        V = V.set(0, s.r / beta)
        H_rows = Tensor.zeros(k, k + 1, backend=backend)

        zero_k1 = Tensor.zeros(k + 1, backend=backend)
        eps: Tensor = Tensor(1e-30, backend=backend)

        def _arnoldi_body(j: Any, carry: tuple) -> tuple:
            V_c, H_rows_c = carry
            w = op.apply(V_c[j])  # matrix-free matvec: (N,)
            h_all = V_c @ w  # (k+1,) inner products
            mask = tensor.arange(k + 1, backend=backend) <= j
            h_masked = tensor.where(mask, h_all, zero_k1)
            w = w - tensor.einsum("i,ij->j", h_masked, V_c)
            h_next = tensor.norm(w)
            v_next = w / (h_next + eps)
            h_row = h_masked.set(j + 1, h_next)
            V_c = V_c.set(j + 1, v_next)
            H_rows_c = H_rows_c.set(j, h_row)
            return V_c, H_rows_c

        V, H_rows = backend.fori_loop(k, _arnoldi_body, (V, H_rows))

        # H̃ = H_rows.T ∈ R^{(k+1)×k}
        H_tilde = tensor.einsum("ij->ji", H_rows)
        g = Tensor.zeros(k + 1, backend=backend)
        g = g.set(0, beta)
        y = SVDFactorization().factorize(H_tilde).solve(g)
        u_new = s.u + tensor.einsum("ij,i->j", V[0:k], y)
        r_new = s.b - op.apply(u_new)
        return _GMRESState(u_new, r_new, s.b, s.iteration + 1)

    def converged(self, state: Any) -> Tensor:
        s: _GMRESState = state
        return (s.r @ s.r < self._tol**2) | (s.iteration >= self._max_iter)

    def extract(self, state: Any) -> Tensor:
        s: _GMRESState = state
        return s.u

    _coverage_predicates = (
        linear_system_predicates()
        + budget_predicates()
        + (
            MembershipPredicate("matrix_representation_available", frozenset({False})),
            MembershipPredicate("linear_operator_matrix_available", frozenset({False})),
            MembershipPredicate("operator_application_available", frozenset({True})),
            ComparisonPredicate("symmetry_defect", ">", LINEARITY_TOLERANCE),
            ComparisonPredicate("singular_value_lower_bound", ">", 0.0),
            ComparisonPredicate("condition_estimate", "<=", CONDITION_LIMIT),
            ComparisonPredicate("rhs_consistency_defect", "<=", LINEARITY_TOLERANCE),
        )
    )

    @classmethod
    def linear_solver_capabilities(cls) -> tuple[LinearSolverCapability, ...]:
        """Return capability declarations owned by this solver implementation."""
        return (
            capability(
                cls,
                contract(
                    requires=(Requirement.LINEAR_OPERATOR, Requirement.NONSINGULAR),
                    provides=(Provision.GENERAL,),
                ),
                coverage_predicates=cls._coverage_predicates,
                coverage_priority=15,
            ),
        )


__all__ = ["DenseGMRESSolver"]
