"""Session-level fixtures for all tests."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from tests.claims import JaxCalibration

# Ensure float64 is enabled before any JAX computation in fixtures.
jax.config.update("jax_enable_x64", True)

_CALIB_N = 100
_CALIB_TRIALS = 20

_JAX_CALIB_N = 256
_JAX_CALIB_WARMUP = 5
_JAX_CALIB_TRIALS = 20


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


def _measure_jax_fma_rate(device: Any) -> float:
    """Return post-JIT JAX FMA rate on the given device in FMAs/second.

    GPU kernel launch overhead (~1 ms) swamps small operations, making dot
    products dispatch-dominated and useless as a GPU roofline.  A 512×512
    matmul (268 M FMAs) is compute-bound at GPU peak throughput so the
    measured rate reflects real GPU performance.  CPU dispatch overhead is
    ~4 µs, so a 256-element dot product (512 FMAs) gives a stable CPU baseline.
    """
    is_gpu = device.platform == "gpu"
    if is_gpu:
        n = 512
        a = jax.device_put(jnp.ones((n, n), dtype=jnp.float64), device)
        b = jax.device_put(jnp.ones((n, n), dtype=jnp.float64), device)
        fn_jit = jax.jit(jnp.matmul)
        fmas_per_call = 2 * n**3
    else:
        n = _JAX_CALIB_N
        a = jax.device_put(jnp.ones(n, dtype=jnp.float64), device)
        b = jax.device_put(jnp.ones(n, dtype=jnp.float64), device)
        fn_jit = jax.jit(jnp.dot)
        fmas_per_call = 2 * n
    for _ in range(_JAX_CALIB_WARMUP):
        fn_jit(a, b).block_until_ready()
    best = float("inf")
    for _ in range(_JAX_CALIB_TRIALS):
        t0 = time.perf_counter()
        fn_jit(a, b).block_until_ready()
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
def jax_calibration() -> JaxCalibration:
    """Session-scoped JAX post-JIT FMA rooflines for CPU and (optionally) GPU.

    cpu_fma_rate is always measured.  gpu_fma_rate is measured when at least
    one GPU device is available AND the measured GPU rate exceeds the CPU rate
    (a GPU slower than CPU indicates a broken driver, e.g. WSL2 without
    proper kernel-mode driver support).  gpu_fma_rate is None otherwise and
    GPU claims skip automatically.
    """
    cpu_device = jax.devices("cpu")[0]
    cpu_rate = _measure_jax_fma_rate(cpu_device)
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    gpu_rate: float | None = None
    if gpu_devices:
        try:
            gpu_rate = _measure_jax_fma_rate(gpu_devices[0])
        except Exception:  # noqa: BLE001
            pass
    return JaxCalibration(cpu_fma_rate=cpu_rate, gpu_fma_rate=gpu_rate)
