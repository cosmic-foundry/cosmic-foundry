"""ProblemDescriptor: numerical parameters that characterize a linear problem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemDescriptor:
    """Numerical parameters that fully characterize a problem for solver selection.

    These four scalar parameters (plus one optional) determine the predicted
    cost of every solver and integrator algorithm, enabling the Autotuner to
    select the fastest configuration without running the full solve.

    Formally: given A u = b with A ∈ ℝⁿˣⁿ, the cost of each algorithm is a
    function of (n, G, r, ε).  The fifth parameter spectral_radius extends
    the same descriptor to time integration du/dt = L(u), where the RHS
    operator L replaces A.

    Parameters
    ----------
    n:
        Matrix (or state) dimension.  Dominant factor in all O(Nᵖ) cost
        models; every algorithm's predicted cost scales as α · Nᵖ.
    g:
        Gershgorin bound on the coefficient matrix A:
          G = max_i Σⱼ |Aᵢⱼ / Aᵢᵢ|
        Proxy for the spectral radius of D⁻¹A (D = diag(A)).  Controls
        Jacobi's relaxation factor ω = min(2/G, 1) and, through the
        contraction rate, the iteration count.  G = 1 means diagonal;
        G ≈ 2 is typical for Poisson stiffness matrices; G > 2 signals
        that ω < 1 (damped Jacobi) is required.  Computed in O(N²) by
        DenseJacobiSolver.init_state — no extra cost beyond what Jacobi
        already pays.
    r:
        Numerical rank of A (integer ≤ n).  Full-rank systems (r = n)
        admit exact LU or Cholesky solutions.  Rank-deficient systems
        (r < n) require SVDFactorization for a minimum-norm pseudoinverse
        solution; LU zero-pinning is a fallback with no norm guarantee.
    tol:
        Solution tolerance ε.  Enters Jacobi's iteration count directly:
        k ≈ log(1/ε) / log(1/ρ) where ρ is the contraction rate.  Does
        not affect direct solvers (LU, Cholesky, SVD), which achieve
        machine precision in a single pass.
    spectral_radius:
        Gershgorin bound on the RHS operator L for time integration.
        Plays the same role as g for solvers: it is the cheap proxy for
        λ_max(L) that determines the CFL-stable step size and hence the
        step count for explicit integrators.  Computed by the same row-sum
        formula as g, applied to the assembled L matrix.  Defaults to 0.0
        when only solver selection is needed.

    In plain terms: n is the problem size, g tells you how hard Jacobi
    will work, r tells you whether the matrix is invertible, tol says how
    precise you need to be, and spectral_radius says how stiff the ODE is
    (for integrator selection).
    """

    n: int
    g: float
    r: int
    tol: float
    spectral_radius: float = 0.0


__all__ = ["ProblemDescriptor"]
