"""Solver convergence verification: monotone residual decrease and tractable count.

When adding a new concrete LinearSolver or a new (solver, discretization) pair
to verify, add a _SolverCase entry to _SOLVER_CASES below.  Each case is
checked by the single parametric test.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.geometry.fvm_discretization import FVMDiscretization
from cosmic_foundry.theory.continuous.dirichlet_bc import DirichletBC
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction

_manifold = EuclideanManifold(1)
_mesh_n8 = CartesianMesh(
    origin=(sympy.Rational(0),),
    spacing=(sympy.Rational(1, 8),),
    shape=(8,),
)


@dataclasses.dataclass
class _SolverCase:
    solver: DenseJacobiSolver
    disc: Any
    rhs: Any
    description: str


_SOLVER_CASES: list[_SolverCase] = [
    _SolverCase(
        solver=DenseJacobiSolver(tol=1e-8, max_iter=10_000),
        disc=FVMDiscretization(
            _mesh_n8,
            DiffusiveFlux(
                DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold
            ),
            DirichletBC(_manifold),
        ),
        rhs=LazyMeshFunction(_mesh_n8, lambda idx: 1.0),
        description="DenseJacobiSolver/DiffusiveFlux(order=4)/N=8",
    ),
]


@pytest.mark.parametrize(
    "case",
    _SOLVER_CASES,
    ids=[c.description for c in _SOLVER_CASES],
)
def test_solver_converges_monotonically(case: _SolverCase) -> None:
    """Solver reaches tol; residuals decrease monotonically; rate < 1.

    Verification (Lane C, C8):
    1. Convergence: final residual < prescribed tol.
    2. Monotone decrease: each iterate reduces ‖f − Au^k‖_{L²_h}.
    3. Empirical spectral radius < 1: geometric mean of the last 20
       per-step ratios (asymptotic rate) is strictly below 1, confirming
       the SPD contraction guarantee holds numerically.
    4. Tractable count: actual iteration count ≤ upper bound derived from
       the asymptotic rate and initial residual.  Because Jacobi damps
       high-frequency modes quickly and slows asymptotically, the tail
       rate is the worst-case per-step ratio; the actual count cannot
       exceed the bound it implies.
    """
    case.solver.solve(case.disc, case.rhs)
    r = case.solver.residuals

    assert r[-1] < case.solver._tol, f"Did not converge: final residual {r[-1]}"

    for k in range(1, len(r)):
        assert (
            r[k] <= r[k - 1]
        ), f"Non-monotone residual at step {k}: {r[k]:.3e} > {r[k - 1]:.3e}"

    # Geometric mean convergence rate over the last min(20, len-1) steps.
    tail_len = min(20, len(r) - 1)
    rho = (r[-1] / r[-1 - tail_len]) ** (1.0 / tail_len)
    assert rho < 1.0, f"Asymptotic rate {rho:.6f} >= 1; solver not contracting"

    # Iteration count ≤ bound implied by the asymptotic rate and initial residual.
    # Tail rate upper-bounds all per-step ratios (Jacobi is fastest early, slowest
    # asymptotically), so r_0 * rho^k_bound >= tol is a valid stopping criterion.
    k_bound = math.ceil(math.log(case.solver._tol / r[0]) / math.log(rho))
    assert (
        len(r) <= k_bound
    ), f"Iteration count {len(r)} exceeds spectral-radius bound {k_bound}"
