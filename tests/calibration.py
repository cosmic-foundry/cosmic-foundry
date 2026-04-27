"""Solver cost-model calibration utilities shared by convergence tests.

_NP_BACKEND and _MESH_FRACTIONS are the shared backend and mesh-refinement
sequence used by all convergence-rate claims.  _calibrate_alpha and
_convergence_n_max select a mesh-size ceiling (N_max) that keeps total
convergence test time within MAX_WALLTIME_S on the current machine.
"""

from __future__ import annotations

import functools
import time
from typing import Any

import sympy

from cosmic_foundry.computation.autotuning.benchmarker import fit_log_log
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.physics.diffusive_flux import DiffusiveFlux
from cosmic_foundry.physics.operator import Operator
from cosmic_foundry.theory.discrete import DirichletGhostCells, FVMDiscretization
from tests.claims import MAX_WALLTIME_S

# NumpyBackend for all convergence claims: numpy SVD/solve are LAPACK-backed
# and the dominant cost is the O(N²) Python-loop assembly, so the cost model
# fits cleanly.  PythonBackend SVD is superlinear at N>64 (cache effects in
# the Jacobi SVD loop), making calibration unreliable there.
_NP_BACKEND = NumpyBackend()

# Mesh fractions: each convergence-rate claim sweeps N_max × f for f in this
# tuple.  The fractions are exact rationals over multiples of 8, so every
# mesh size is an integer whenever N_max is a multiple of 8.
_MESH_FRACTIONS = (0.25, 0.375, 0.5, 0.75, 1.0)

_CALIB_N = 64  # mesh size used to calibrate each solver's cost coefficient

_manifold = EuclideanManifold(1)


def _time_solve_at(solver_class: type, n: int) -> float:
    """Return the best-of-3 wall time (seconds) for assemble + svd + solve at size n.

    Uses NumpyBackend so that SVD and solve are fast (LAPACK-backed) and the
    dominant cost is the O(N²) Python-loop assembly.  The convergence claims use
    the same backend, so calibration and tests measure the same code paths.
    """
    mesh = CartesianMesh(
        origin=(sympy.Rational(0),),
        spacing=(sympy.Rational(1, n),),
        shape=(n,),
    )
    flux = DiffusiveFlux(DiffusiveFlux.min_order, _manifold)
    disc = FVMDiscretization(mesh, flux, DirichletGhostCells())
    a_cal = Operator(disc(), mesh).assemble(backend=_NP_BACKEND)
    b_cal = Tensor([1.0] * n, backend=_NP_BACKEND)
    solver = solver_class()
    solver.solve(a_cal, b_cal)  # warm-up: ensure any lazy initialization is done
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        a_cal = Operator(disc(), mesh).assemble(backend=_NP_BACKEND)
        a_cal.svd()
        solver.solve(a_cal, b_cal)
        best = min(best, time.perf_counter() - t0)
    return best


@functools.cache
def _calibrate_alpha(solver_class: type, fma_rate: float) -> tuple[float, float]:
    """Empirically fit (alpha, exponent) for solver_class from a two-point log-log fit.

    The cost model is T ≈ alpha × N^exponent / fma_rate.  Both alpha and
    exponent are measured rather than declared: time at _CALIB_N // 2 and
    _CALIB_N, fit exponent from the log-log slope, then pin alpha at _CALIB_N.
    Including assembly and SVD in the timing ensures N_max is conservative enough
    that all three phases of each convergence claim fit within the walltime budget.
    Memoised on (solver_class, fma_rate) so each (solver type, machine) pair
    calibrates once per session.
    """
    n1 = _CALIB_N // 2
    n2 = _CALIB_N
    t1 = _time_solve_at(solver_class, n1)
    t2 = _time_solve_at(solver_class, n2)
    alpha_raw, exponent = fit_log_log([(n1, t1), (n2, t2)])
    alpha = alpha_raw * fma_rate
    return alpha, exponent


def _convergence_n_max(fma_rate: float, n_convergence_claims: int, solver: Any) -> int:
    """N_max for the convergence mesh sequence for solver given the machine's FMA rate.

    Allocates MAX_WALLTIME_S equally across all convergence-rate claims and
    solves for the N_max each claim can afford under its solver's cost model
    T ≈ alpha × N^p × Σ(f^p) / fma_rate.  alpha and p are calibrated once per
    (solver type, fma_rate) pair by _calibrate_alpha.  Rounding to the nearest
    multiple of 8 keeps all mesh sizes exact integers.
    """
    alpha, p = _calibrate_alpha(type(solver), fma_rate)
    sum_fp = sum(f**p for f in _MESH_FRACTIONS)
    budget_per_claim = MAX_WALLTIME_S / n_convergence_claims
    n_raw = (budget_per_claim * fma_rate / (alpha * sum_fp)) ** (1 / p)
    return max(16, round(n_raw / 8) * 8)
