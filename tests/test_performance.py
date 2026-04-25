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
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import pytest

from cosmic_foundry.computation.tensor import Tensor

# Regressions larger than this multiple of the roofline prediction fail.
EFFICIENCY_FACTOR = 8

# Number of trials; the minimum time across trials is used to eliminate
# OS scheduling noise while still catching algorithmic slowdowns.
_TRIALS = 20


def _measure_fma_rate() -> float:
    """Return the pure-Python list FMA rate in FMAs/second.

    Runs the same inner loop as Tensor._matvec — list-element
    multiply-accumulate — and returns the peak rate across _TRIALS
    repetitions.  Using the maximum rate (minimum elapsed time) gives
    the tightest roofline: a factor-of-N regression shows up as the
    measured time exceeding EFFICIENCY_FACTOR × (flops / fma_rate).
    """
    n = 100
    a = [float(i) * 0.001 + 1.0 for i in range(n)]
    b = [float(i) * 0.001 + 1.0 for i in range(n)]
    best_elapsed = float("inf")
    for _ in range(_TRIALS):
        t0 = time.perf_counter()
        _ = sum(a[i] * b[i] for i in range(n))
        best_elapsed = min(best_elapsed, time.perf_counter() - t0)
    return n / best_elapsed


@pytest.fixture(scope="session")
def fma_rate() -> float:
    """Session-scoped FMA roofline: pure-Python list FMAs per second."""
    return _measure_fma_rate()


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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CLAIMS: list[_PerfClaim] = [
    *[_DotPerfClaim(n) for n in [8, 32, 128]],
    *[_MatvecPerfClaim(n) for n in [8, 16, 32]],
    *[_MatmulPerfClaim(n) for n in [8, 16]],
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_performance(claim: _PerfClaim, fma_rate: float) -> None:
    claim.check(fma_rate)
