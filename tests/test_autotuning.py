"""Autotuning claims.

Claim types:
  _SelectionValidClaim   — selected solver produces a valid solution on a
                           full-rank problem at calibration size
  _CrossoverClaim        — at large N, the autotuner selects the lower-exponent
                           solver (CG, p=2) over the higher-exponent solver
                           (LU, p=3), whether via screening or cost model
"""

from __future__ import annotations

import pytest
import sympy

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.autotuning.autotuner import Autotuner
from cosmic_foundry.computation.autotuning.benchmarker import Benchmarker
from cosmic_foundry.computation.autotuning.problem_descriptor import ProblemDescriptor
from cosmic_foundry.computation.backends import Backend, NumpyBackend
from cosmic_foundry.computation.solvers.dense_cg_solver import DenseCGSolver
from cosmic_foundry.computation.solvers.dense_lu_solver import DenseLUSolver
from cosmic_foundry.computation.solvers.linear_solver import LinearOperator
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.discrete import (
    DiffusiveFlux,
    DirichletGhostCells,
    DivergenceFormDiscretization,
)
from tests.claims import Claim, assemble_linear_op

# Calibration size: small enough to be cheap, large enough that the power-law
# extrapolation to _LARGE_N is meaningful.
_CALIB_N = 16
_LARGE_N = 128

# Poisson-like descriptor at calibration size: G = 2 (tridiagonal stiffness),
# full rank, standard convergence tolerance.
_CALIB_DESCRIPTOR = ProblemDescriptor(n=_CALIB_N, g=2.0, r=_CALIB_N, tol=1e-8)
_LARGE_DESCRIPTOR = ProblemDescriptor(n=_LARGE_N, g=2.0, r=_LARGE_N, tol=1e-8)

_BACKEND = NumpyBackend()
_MANIFOLD = EuclideanManifold(1)
# CG: O(N²) with SpMV — O(N) iterations each doing one O(N) apply.
# LU: O(N³) — N SpMV calls for assembly plus LAPACK factorization.
_SOLVERS = [DenseLUSolver(), DenseCGSolver()]


def _op_factory(n: int, backend: Backend) -> tuple[LinearOperator, Tensor]:
    """Construct a 1-D Poisson FVM operator of size n on the given backend."""
    mesh = CartesianMesh(
        origin=(sympy.Rational(0),),
        spacing=(sympy.Rational(1, n),),
        shape=(n,),
    )
    flux = DiffusiveFlux(DiffusiveFlux.min_order, _MANIFOLD)
    disc = DivergenceFormDiscretization(flux, DirichletGhostCells())
    op: LinearOperator = assemble_linear_op(disc, mesh)
    b = Tensor([1.0] * n, backend=backend)
    return op, b


class _SelectionValidClaim(Claim[None]):
    """Claim: the selected solver produces ‖Au − b‖₂ < tol on a full-rank problem."""

    @property
    def description(self) -> str:
        return "autotuning/selection_valid"

    def check(self, _calibration: None) -> None:
        autotuner = Autotuner(
            _SOLVERS,
            [_BACKEND],
            operator_factory=_op_factory,
            benchmarker=Benchmarker(n_trials=3),
        )
        autotuner.calibrate(_CALIB_DESCRIPTOR)
        selection = autotuner.select(_CALIB_DESCRIPTOR)

        op, b = _op_factory(_CALIB_N, _BACKEND)
        u = selection.solver.solve(op, b)
        residual = tensor.norm(b - op.apply(u))
        assert residual.get() < _CALIB_DESCRIPTOR.tol, (
            f"selected {type(selection.solver).__name__} residual "
            f"{residual.get():.2e} >= tol {_CALIB_DESCRIPTOR.tol}"
        )


class _CrossoverClaim(Claim[None]):
    """Claim: at large N, the autotuner selects DenseCGSolver (p=2) over LU (p=3).

    With a sparse FVM operator, CG costs O(N²): O(N) iterations each doing one
    O(N) SpMV.  DirectSolver (LU) costs O(N³): N SpMV calls for assembly plus
    LAPACK factorization.  At N = 128 and beyond, the N² vs N³ gap dominates
    any plausible difference in α coefficients.
    """

    @property
    def description(self) -> str:
        return "autotuning/cg_beats_lu_at_large_n"

    def check(self, _calibration: None) -> None:
        autotuner = Autotuner(
            _SOLVERS,
            [_BACKEND],
            operator_factory=_op_factory,
            benchmarker=Benchmarker(n_trials=3),
        )
        autotuner.calibrate(_CALIB_DESCRIPTOR)
        selection = autotuner.select(_LARGE_DESCRIPTOR)

        assert isinstance(selection.solver, DenseCGSolver), (
            f"expected DenseCGSolver at N={_LARGE_N}, "
            f"got {type(selection.solver).__name__} "
            f"(predicted cost {selection.predicted_cost:.2e}s)"
        )


_CLAIMS: list[Claim[None]] = [
    _SelectionValidClaim(),
    _CrossoverClaim(),
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=lambda c: c.description)
def test_autotuning(claim: Claim[None]) -> None:
    claim.check(None)
