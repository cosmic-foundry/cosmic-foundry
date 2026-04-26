"""Session-level fixtures for all tests."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from cosmic_foundry.computation.backends import JaxBackend
from cosmic_foundry.computation.tensor import Tensor
from tests.claims import (
    BUDGET_TOLERANCE,
    MAX_WALLTIME_S,
    ConvergenceCalibration,
    DeviceCalibration,
)


def pytest_configure(config: pytest.Config) -> None:
    """Set the pytest-timeout session backstop from the shared budget constants.

    Sets the session timeout from MAX_WALLTIME_S × BUDGET_TOLERANCE so that
    changes to tests/claims.py update the timeout automatically without
    requiring command-line flags.
    """
    timeout = MAX_WALLTIME_S * BUDGET_TOLERANCE
    # Set the option so pytest-timeout reads it
    config.option.session_timeout = timeout
    # Store for use in pytest_sessionstart
    config._cosmic_foundry_session_timeout_seconds = timeout


def pytest_sessionstart(session: pytest.Session) -> None:
    """Start a background thread that enforces the session timeout.

    This ensures the process exits hard if the session exceeds
    MAX_WALLTIME_S × BUDGET_TOLERANCE, even if a test is running.
    """
    timeout_seconds = session.config._cosmic_foundry_session_timeout_seconds

    def _timeout_monitor() -> None:
        """Background thread that kills the process on timeout."""
        time.sleep(timeout_seconds)
        # Session timeout exceeded; exit hard
        os._exit(1)

    thread = threading.Thread(target=_timeout_monitor, daemon=True)
    thread.start()


_CALIB_N = 100
_CALIB_TRIALS = 20

_JACOBI_CALIB_N = 32
_JACOBI_CALIB_TRIALS = 3

_DEVICE_CALIB_N_CPU = 256
_DEVICE_CALIB_N_GPU = 512
_DEVICE_CALIB_WARMUP = 5
_DEVICE_CALIB_TRIALS = 20


def _measure_fma_rate() -> float:
    """Return pure-Python list FMA rate in FMAs/second.

    Times the same inner loop pattern used by Tensor._matvec (element-wise
    multiply-accumulate over a list) and returns the peak rate across
    _CALIB_TRIALS repetitions.  Taking the minimum elapsed time eliminates OS
    scheduling noise while still catching algorithmic slowdowns.

    Used only by PythonBackend performance claims.  Convergence-rate mesh-size
    selection uses convergence_calibration, which carries device-specific Jacobi
    cost coefficients calibrated by real timed solves on each device.
    """
    n = _CALIB_N
    a = [float(i) * 0.001 + 1.0 for i in range(n)]
    b = [float(i) * 0.001 + 1.0 for i in range(n)]
    best_elapsed = float("inf")
    for _ in range(_CALIB_TRIALS):
        t0 = time.perf_counter()
        _ = sum(a[i] * b[i] for i in range(n))
        best_elapsed = min(best_elapsed, time.perf_counter() - t0)
    return n / best_elapsed


def _measure_backend_fma_rate(backend: JaxBackend, is_gpu: bool) -> float:
    """Return FMA throughput for backend in FMAs/second, using Tensor ops.

    GPU kernel launch overhead (~1 ms) swamps small operations, so a 512×512
    matmul (268 M FMAs) is used for GPU to ensure the measurement is
    compute-bound.  CPU dispatch overhead is ~4 µs, so a 256-element dot
    product (512 FMAs) is sufficient for the CPU baseline.

    Warmup calls allow the backend's dispatch cache (e.g. XLA compilation) to
    stabilise before timing begins.  Tensor.sync() blocks until async
    dispatches complete so GPU timing is accurate.
    """
    if is_gpu:
        n = _DEVICE_CALIB_N_GPU
        a = Tensor(np.ones((n, n)), backend=backend)
        b = Tensor(np.ones((n, n)), backend=backend)
        fmas_per_call = 2 * n**3
    else:
        n = _DEVICE_CALIB_N_CPU
        a = Tensor(np.ones(n), backend=backend)
        b = Tensor(np.ones(n), backend=backend)
        fmas_per_call = 2 * n
    for _ in range(_DEVICE_CALIB_WARMUP):
        r = a @ b
        r.sync()
    best = float("inf")
    for _ in range(_DEVICE_CALIB_TRIALS):
        t0 = time.perf_counter()
        r = a @ b
        r.sync()
        best = min(best, time.perf_counter() - t0)
    return fmas_per_call / best


@pytest.fixture(scope="session")
def fma_rate() -> float:
    """Session-scoped FMA roofline: pure-Python list FMAs per second.

    Calibrated once at session start.  Performance claims and convergence
    mesh-size selection both consume this fixture rather than running their
    own timing loops.
    """
    return _measure_fma_rate()


@pytest.fixture(scope="session")
def device_calibration() -> DeviceCalibration:
    """Session-scoped FMA rooflines for CPU and (optionally) GPU.

    cpu_fma_rate is measured using a CPU JaxBackend and a dot-product workload
    (dispatch-limited baseline).  gpu_fma_rate is measured using a GPU
    JaxBackend and a compute-bound matmul workload; it is None when no GPU
    device is available or when the XLA driver raises during measurement.

    The backend instances stored in the result are the same ones used during
    calibration; performance claims use them so the benchmark runs through
    identical code paths.
    """
    cpu_backend = JaxBackend()
    cpu_rate = _measure_backend_fma_rate(cpu_backend, is_gpu=False)
    gpu_backend: JaxBackend | None = None
    gpu_rate: float | None = None
    try:
        gpu_backend = JaxBackend(device="gpu")
        gpu_rate = _measure_backend_fma_rate(gpu_backend, is_gpu=True)
    except Exception:  # noqa: BLE001
        gpu_backend = None
    return DeviceCalibration(
        cpu_backend=cpu_backend,
        gpu_backend=gpu_backend,
        cpu_fma_rate=cpu_rate,
        gpu_fma_rate=gpu_rate,
    )


def _calibrate_solver_alpha(solver: object, backend: object) -> float:
    """Return cost coefficient alpha for solver on backend, calibrated by timing.

    Times solver.solve on a _JACOBI_CALIB_N DiffusiveFlux problem assembled on
    backend and returns alpha = T_best / N^p where p = solver.cost_exponent.
    Predicted time at mesh size M is then alpha · M^p, used by
    _convergence_n_max to size each claim's mesh sequence.

    Calibration uses the same solver instance the test will run, so the alpha
    captures every cost source: kernel-launch overhead, dispatch latency,
    memory bandwidth, and the solver's iteration count or factorization style.
    """
    import sympy

    from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
    from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
    from cosmic_foundry.physics.diffusive_flux import DiffusiveFlux
    from cosmic_foundry.physics.fvm_discretization import FVMDiscretization
    from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
        DirichletGhostCells,
    )

    n = _JACOBI_CALIB_N
    manifold = EuclideanManifold(1)
    mesh = CartesianMesh(
        origin=(sympy.Rational(0),),
        spacing=(sympy.Rational(1, n),),
        shape=(n,),
    )
    flux = DiffusiveFlux(DiffusiveFlux.min_order, manifold)
    disc = FVMDiscretization(mesh, flux, DirichletGhostCells())
    a_cal = disc.assemble(backend=backend)
    b_cal = Tensor([1.0] * n, backend=backend)
    r = solver.solve(a_cal, b_cal)  # warm-up: let XLA compile
    r.sync()
    best = float("inf")
    for _ in range(_JACOBI_CALIB_TRIALS):
        t0 = time.perf_counter()
        r = solver.solve(a_cal, b_cal)
        r.sync()
        best = min(best, time.perf_counter() - t0)
    return best / n**solver.cost_exponent


def _calibrate_solver_alphas(backend: object) -> dict[type, float]:
    """Calibrate every LinearSolver class registered for convergence testing."""
    from cosmic_foundry.computation.dense_jacobi_solver import DenseJacobiSolver
    from cosmic_foundry.computation.dense_lu_solver import DenseLUSolver

    return {
        DenseJacobiSolver: _calibrate_solver_alpha(
            DenseJacobiSolver(tol=1e-8), backend
        ),
        DenseLUSolver: _calibrate_solver_alpha(DenseLUSolver(), backend),
    }


@pytest.fixture(scope="session")
def convergence_calibration(
    device_calibration: DeviceCalibration,
) -> ConvergenceCalibration:
    """Session-scoped per-(device, solver) calibration for convergence claims.

    Calibrates each registered LinearSolver class on each available device by
    running a real timed solve.  The resulting α values let _convergence_n_max
    size each claim's mesh sequence correctly: an O(N^3) LU claim sees a much
    larger N_max than an O(N^4) Jacobi claim for the same time budget.
    """
    cpu_alphas = _calibrate_solver_alphas(device_calibration.cpu_backend)
    gpu_alphas: dict[type, float] | None = None
    if device_calibration.gpu_backend is not None:
        gpu_alphas = _calibrate_solver_alphas(device_calibration.gpu_backend)
    return ConvergenceCalibration(
        cpu_backend=device_calibration.cpu_backend,
        gpu_backend=device_calibration.gpu_backend,
        cpu_alphas=cpu_alphas,
        gpu_alphas=gpu_alphas,
    )
