"""Performance regression guard for Tensor operations.

Each claim verifies that a Tensor operation completes within
EFFICIENCY_FACTOR of the machine's measured pure-Python FMA roofline.

The roofline is calibrated at session start by timing a reference
dot-product loop — the same code pattern used internally by
Tensor.__matmul__.  This makes every bound self-calibrating: the test
passes on a slow CI runner and on a fast developer workstation alike,
while still catching algorithmic regressions.

An efficiency factor of 8 means a regression that makes any operation
more than 8× slower than the roofline predicts will fail the test.
The einsum regression that motivated this suite was ~15×, well above
that threshold.

_NumpyParityPerfClaim bounds the overhead of the NumpyBackend Tensor
wrapper relative to raw NumPy: at most NUMPY_PARITY_FACTOR = 2 for
representative workloads (N ≥ 8 matmul and matvec).

_BackendSpeedupClaim bounds the minimum speedup NumpyBackend achieves
over PythonBackend, catching regressions where NumPy is accidentally
bypassed or replaced by pure-Python fallback code.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import pytest

from cosmic_foundry.computation.backends import NumpyBackend, PythonBackend
from cosmic_foundry.computation.tensor import Tensor

# Regressions larger than this multiple of the roofline prediction fail.
EFFICIENCY_FACTOR = 8

# NumpyBackend Tensor must stay within this multiple of raw NumPy throughput.
NUMPY_PARITY_FACTOR = 2

# Number of trials; the minimum time across trials is used to eliminate
# OS scheduling noise while still catching algorithmic slowdowns.
_TRIALS = 20

_PY = PythonBackend()
_NP = NumpyBackend()


# ---------------------------------------------------------------------------
# Claim base class and concrete types
# ---------------------------------------------------------------------------


class _PerfClaim(ABC):
    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self, fma_rate: float) -> None: ...


class _MatvecPerfClaim(_PerfClaim):
    """Claim: N×N @ N matvec runs within EFFICIENCY_FACTOR of the FMA roofline.

    Expected FMAs: 2N² (N rows × N multiply-adds).
    """

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def description(self) -> str:
        return f"matvec/N={self._n}"

    def check(self, fma_rate: float) -> None:
        n = self._n
        a = Tensor([[float(i + j) for j in range(n)] for i in range(n)])
        x = Tensor([float(i) for i in range(n)])

        best_elapsed = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            _ = a @ x
            best_elapsed = min(best_elapsed, time.perf_counter() - t0)

        expected = 2 * n**2 / fma_rate
        assert best_elapsed <= EFFICIENCY_FACTOR * expected, (
            f"matvec N={n}: {best_elapsed * 1e6:.1f}µs actual, "
            f"{expected * 1e6:.1f}µs roofline, "
            f"{best_elapsed / expected:.1f}× > {EFFICIENCY_FACTOR}× limit"
        )


class _MatmulPerfClaim(_PerfClaim):
    """Claim: N×N @ N×N matmul runs within EFFICIENCY_FACTOR of the FMA roofline.

    Expected FMAs: 2N³ (N² output elements × N multiply-adds each).
    """

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def description(self) -> str:
        return f"matmul/N={self._n}"

    def check(self, fma_rate: float) -> None:
        n = self._n
        a = Tensor([[float(i + j) for j in range(n)] for i in range(n)])
        b = Tensor([[float(i * j + 1) for j in range(n)] for i in range(n)])

        best_elapsed = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            _ = a @ b
            best_elapsed = min(best_elapsed, time.perf_counter() - t0)

        expected = 2 * n**3 / fma_rate
        assert best_elapsed <= EFFICIENCY_FACTOR * expected, (
            f"matmul N={n}: {best_elapsed * 1e6:.1f}µs actual, "
            f"{expected * 1e6:.1f}µs roofline, "
            f"{best_elapsed / expected:.1f}× > {EFFICIENCY_FACTOR}× limit"
        )


class _DotPerfClaim(_PerfClaim):
    """Claim: N @ N dot product runs within EFFICIENCY_FACTOR of the FMA roofline.

    Expected FMAs: 2N (N multiplies + N-1 adds ≈ 2N).
    """

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def description(self) -> str:
        return f"dot/N={self._n}"

    def check(self, fma_rate: float) -> None:
        n = self._n
        a = Tensor([float(i) for i in range(n)])
        b = Tensor([float(i) * 0.5 for i in range(n)])

        best_elapsed = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            _ = a @ b
            best_elapsed = min(best_elapsed, time.perf_counter() - t0)

        expected = 2 * n / fma_rate
        assert best_elapsed <= EFFICIENCY_FACTOR * expected, (
            f"dot N={n}: {best_elapsed * 1e6:.1f}µs actual, "
            f"{expected * 1e6:.1f}µs roofline, "
            f"{best_elapsed / expected:.1f}× > {EFFICIENCY_FACTOR}× limit"
        )


class _NumpyParityPerfClaim(_PerfClaim):
    """NumpyBackend Tensor op ≤ NUMPY_PARITY_FACTOR × raw NumPy op.

    Measures the overhead of the Tensor wrapper (backend dispatch, shape
    inference, _wrap) relative to calling NumPy directly.  Both the raw
    NumPy arrays and the NumpyBackend Tensors are constructed before the
    timed loops so that construction cost is excluded.

    op must be one of "matmul" (N×N @ N×N) or "matvec" (N×N @ N).
    """

    def __init__(self, op: str, n: int) -> None:
        self._op = op
        self._n = n

    @property
    def description(self) -> str:
        return f"numpy_parity/{self._op}/N={self._n}"

    def check(self, fma_rate: float) -> None:
        n = self._n
        raw_a = np.array([[float(i + j) for j in range(n)] for i in range(n)])
        ta = Tensor([[float(i + j) for j in range(n)] for i in range(n)], backend=_NP)

        if self._op == "matmul":
            raw_b = np.array([[float(i * j + 1) for j in range(n)] for i in range(n)])
            tb = Tensor(
                [[float(i * j + 1) for j in range(n)] for i in range(n)], backend=_NP
            )

            def np_op() -> None:
                np.matmul(raw_a, raw_b)

            def tensor_op() -> None:
                ta @ tb

        else:
            raw_x = np.array([float(i) for i in range(n)])
            tx = Tensor([float(i) for i in range(n)], backend=_NP)

            def np_op() -> None:
                np.matmul(raw_a, raw_x)

            def tensor_op() -> None:
                ta @ tx

        best_np = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            np_op()
            best_np = min(best_np, time.perf_counter() - t0)

        best_tensor = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            tensor_op()
            best_tensor = min(best_tensor, time.perf_counter() - t0)

        assert best_tensor <= NUMPY_PARITY_FACTOR * best_np, (
            f"{self.description}: "
            f"Tensor={best_tensor * 1e6:.2f}µs  "
            f"NumPy={best_np * 1e6:.2f}µs  "
            f"ratio={best_tensor / best_np:.2f}× > {NUMPY_PARITY_FACTOR}× limit"
        )


class _BackendSpeedupClaim(_PerfClaim):
    """NumpyBackend Tensor op is at least min_speedup× faster than PythonBackend.

    Catches regressions where NumPy is accidentally bypassed (e.g. an
    operation falls back to pure-Python loops).  The minimum speedup is
    set conservatively below the observed speedup so that natural variation
    in timing does not produce false failures.

    op must be one of "matmul" (N×N @ N×N) or "matvec" (N×N @ N).
    """

    def __init__(self, op: str, n: int, min_speedup: int) -> None:
        self._op = op
        self._n = n
        self._min_speedup = min_speedup

    @property
    def description(self) -> str:
        return f"numpy_speedup/{self._op}/N={self._n}"

    def check(self, fma_rate: float) -> None:
        n = self._n
        py_a = Tensor([[float(i + j) for j in range(n)] for i in range(n)], backend=_PY)
        np_a = Tensor([[float(i + j) for j in range(n)] for i in range(n)], backend=_NP)

        if self._op == "matmul":
            py_b = Tensor(
                [[float(i * j + 1) for j in range(n)] for i in range(n)], backend=_PY
            )
            np_b = Tensor(
                [[float(i * j + 1) for j in range(n)] for i in range(n)], backend=_NP
            )

            def py_op() -> None:
                py_a @ py_b

            def np_op() -> None:
                np_a @ np_b

        else:
            py_x = Tensor([float(i) for i in range(n)], backend=_PY)
            np_x = Tensor([float(i) for i in range(n)], backend=_NP)

            def py_op() -> None:
                py_a @ py_x

            def np_op() -> None:
                np_a @ np_x

        best_py = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            py_op()
            best_py = min(best_py, time.perf_counter() - t0)

        best_np = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            np_op()
            best_np = min(best_np, time.perf_counter() - t0)

        speedup = best_py / best_np
        assert speedup >= self._min_speedup, (
            f"{self.description}: "
            f"Python={best_py * 1e6:.1f}µs  "
            f"NumPy={best_np * 1e6:.2f}µs  "
            f"speedup={speedup:.1f}× < {self._min_speedup}× minimum"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CLAIMS: list[_PerfClaim] = [
    # PythonBackend vs FMA roofline
    *[_DotPerfClaim(n) for n in [8, 32, 128]],
    *[_MatvecPerfClaim(n) for n in [8, 16, 32]],
    *[_MatmulPerfClaim(n) for n in [8, 16]],
    # NumpyBackend vs raw NumPy: wrapper overhead ≤ 2×
    *[_NumpyParityPerfClaim("matmul", n) for n in [8, 16, 32]],
    *[_NumpyParityPerfClaim("matvec", n) for n in [8, 16, 32]],
    # NumpyBackend vs PythonBackend: NumPy must be faster by at least min_speedup
    *[_BackendSpeedupClaim("matmul", n, 10) for n in [8, 16, 32]],
    *[_BackendSpeedupClaim("matvec", n, 5) for n in [16, 32]],
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_performance(claim: _PerfClaim, fma_rate: float) -> None:
    claim.check(fma_rate)
