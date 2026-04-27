"""Autotuning claims.

Claim types:
  _SelectionValidClaim   — selected solver produces a valid solution on a
                           full-rank problem at calibration size
  _CrossoverClaim        — at large N, the lower-exponent solver (LU, p=3)
                           is predicted faster than the higher-exponent
                           solver (Jacobi, p=4)
"""

from __future__ import annotations

import pytest

from cosmic_foundry.computation.autotuning.autotuner import Autotuner
from cosmic_foundry.computation.autotuning.benchmarker import Benchmarker
from cosmic_foundry.computation.autotuning.problem_descriptor import ProblemDescriptor
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.solvers.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.solvers.dense_lu_solver import DenseLUSolver
from cosmic_foundry.computation.tensor import Tensor
from tests.claims import Claim

# Calibration size: small enough to be cheap, large enough that the power-law
# extrapolation to _LARGE_N is meaningful.
_CALIB_N = 16
_LARGE_N = 128

# Poisson-like descriptor at calibration size: G = 2 (tridiagonal stiffness),
# full rank, standard convergence tolerance.
_CALIB_DESCRIPTOR = ProblemDescriptor(n=_CALIB_N, g=2.0, r=_CALIB_N, tol=1e-8)
_LARGE_DESCRIPTOR = ProblemDescriptor(n=_LARGE_N, g=2.0, r=_LARGE_N, tol=1e-8)

_BACKEND = NumpyBackend()
_SOLVERS = [DenseLUSolver(), DenseJacobiSolver(tol=1e-8)]


class _SelectionValidClaim(Claim):
    """Claim: the selected solver produces ‖Au − b‖₂ < tol on a full-rank problem."""

    @property
    def description(self) -> str:
        return "autotuning/selection_valid"

    def check(self) -> None:
        autotuner = Autotuner(_SOLVERS, [_BACKEND], benchmarker=Benchmarker(n_trials=3))
        autotuner.calibrate(_CALIB_DESCRIPTOR)
        selection = autotuner.select(_CALIB_DESCRIPTOR)

        # Build the same synthetic matrix used for calibration and verify solve.
        a = Benchmarker._make_matrix(_CALIB_DESCRIPTOR, _BACKEND)
        b = Tensor([1.0] * _CALIB_N, backend=_BACKEND)
        u = selection.solver.solve(a, b)
        residual = (b - a @ u).norm()
        assert residual.get() < _CALIB_DESCRIPTOR.tol, (
            f"selected {type(selection.solver).__name__} residual "
            f"{residual.get():.2e} >= tol {_CALIB_DESCRIPTOR.tol}"
        )


class _CrossoverClaim(Claim):
    """Claim: at large N, the autotuner predicts LU (p=3) faster than Jacobi (p=4).

    At N = 128 and beyond, the N³ vs N⁴ exponent gap dominates any plausible
    difference in α coefficients.  A failure here would mean DenseLUSolver's
    measured α is implausibly large compared to DenseJacobiSolver's — a signal
    that the cost model or benchmarking is broken rather than that Jacobi is
    genuinely faster.
    """

    @property
    def description(self) -> str:
        return "autotuning/lu_beats_jacobi_at_large_n"

    def check(self) -> None:
        autotuner = Autotuner(_SOLVERS, [_BACKEND], benchmarker=Benchmarker(n_trials=3))
        autotuner.calibrate(_CALIB_DESCRIPTOR)

        # Extract the alpha for each solver from calibration results.
        alphas = {type(r.solver): r.alpha for r in autotuner.results}
        lu_cost = alphas[DenseLUSolver] * _LARGE_N**DenseLUSolver.cost_exponent
        jacobi_cost = (
            alphas[DenseJacobiSolver] * _LARGE_N**DenseJacobiSolver.cost_exponent
        )

        assert lu_cost < jacobi_cost, (
            f"expected LU (predicted {lu_cost:.2e}s) < Jacobi (predicted "
            f"{jacobi_cost:.2e}s) at N={_LARGE_N}, but Jacobi was predicted faster. "
            f"LU α={alphas[DenseLUSolver]:.2e}, "
            f"Jacobi α={alphas[DenseJacobiSolver]:.2e}"
        )


_CLAIMS: list[Claim] = [
    _SelectionValidClaim(),
    _CrossoverClaim(),
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=lambda c: c.description)
def test_autotuning(claim: Claim) -> None:
    claim.check()
