"""Session-level fixtures for all tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from cosmic_foundry.computation.backends import JaxBackend
from cosmic_foundry.computation.tensor import Tensor
from tests.claims import DeviceCalibration

_CALIB_N = 100
_CALIB_TRIALS = 20

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

    When the active solver backend is accelerated (NumPy, JAX, GPU), update
    this function to measure the backend's throughput instead of pure-Python
    speed, so that convergence-sweep mesh-size selection tracks the actual
    solver rate.
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
