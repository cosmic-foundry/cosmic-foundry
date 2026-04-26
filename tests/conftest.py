"""Session-level fixtures for all tests."""

from __future__ import annotations

import time

import pytest

_CALIB_N = 100
_CALIB_TRIALS = 20


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


@pytest.fixture(scope="session")
def fma_rate() -> float:
    """Session-scoped FMA roofline: pure-Python list FMAs per second.

    Calibrated once at session start.  Performance claims and convergence
    mesh-size selection both consume this fixture rather than running their
    own timing loops.
    """
    return _measure_fma_rate()
