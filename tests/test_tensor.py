"""Tensor verification claims for all backends.

Correctness claims encode Tensor / Backend semantic properties. Performance
claims encode Tensor/backend roofline and parity checks. Adding a claim requires
only appending it to the appropriate axis registry.

Claim types:
  _RoundtripClaim(backend)       — to_native → from_native preserves values
  _ArithmeticClaim(backend, op)  — arithmetic result matches PythonBackend reference
  _ConversionClaim(src, dst)     — .to() preserves shape and element values
  _MixedBackendClaim(op)         — mixing backends raises ValueError
  _FactoryClaim(backend, name)   — Tensor.zeros / Tensor.eye use the given backend
  _SliceClaim(backend, op)       — slice read/write matches PythonBackend reference
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.backends import JaxBackend, NumpyBackend, PythonBackend
from cosmic_foundry.computation.tensor import Tensor, arange, einsum, where
from tests.claims import Claim, DeviceCalibration

_PY = PythonBackend()
_NP = NumpyBackend()
_JAX = JaxBackend()

_NON_PY_BACKENDS = (_NP, _JAX)


# ---------------------------------------------------------------------------
# Round-trip: to_native → from_native leaves values unchanged
# ---------------------------------------------------------------------------


class _RoundtripClaim(Claim[None]):
    def __init__(self, backend: Any, label: str, data: Any) -> None:
        self._backend = backend
        self._label = label
        self._data = data

    @property
    def description(self) -> str:
        return f"roundtrip/{self._label}/{type(self._backend).__name__}"

    def check(self, _calibration: None) -> None:
        t = Tensor(self._data, backend=self._backend)
        recovered = t.to_list()
        ref = Tensor(self._data, backend=_PY).to_list()
        assert _approx_equal(
            recovered, ref
        ), f"{self.description}: got {recovered!r}, expected {ref!r}"


# ---------------------------------------------------------------------------
# Arithmetic: backend result matches PythonBackend reference
# ---------------------------------------------------------------------------


class _ArithmeticClaim(Claim[None]):
    def __init__(self, label: str, fn: Any, backend: Any = None) -> None:
        self._label = label
        self._fn = fn
        self._backend = _NP if backend is None else backend

    @property
    def description(self) -> str:
        return f"arithmetic/{self._label}/{type(self._backend).__name__}"

    def check(self, _calibration: None) -> None:
        py_result = self._fn(_PY)
        test_result = self._fn(self._backend)
        assert _approx_equal(py_result.to_list(), test_result.to_list()), (
            f"{self.description}: PythonBackend={py_result.to_list()!r} "
            f"{type(self._backend).__name__}={test_result.to_list()!r}"
        )


# ---------------------------------------------------------------------------
# Conversion: .to() preserves shape and values across backends
# ---------------------------------------------------------------------------


class _ConversionClaim(Claim[None]):
    def __init__(self, src: Any, dst: Any, label: str, data: Any) -> None:
        self._src = src
        self._dst = dst
        self._label = label
        self._data = data

    @property
    def description(self) -> str:
        return (
            f"conversion/{self._label}/"
            f"{type(self._src).__name__}→{type(self._dst).__name__}"
        )

    def check(self, _calibration: None) -> None:
        original = Tensor(self._data, backend=self._src)
        converted = original.to(self._dst)
        assert (
            converted.shape == original.shape
        ), f"{self.description}: shape {converted.shape} != {original.shape}"
        assert _approx_equal(
            converted.to_list(), original.to_list()
        ), f"{self.description}: values differ after .to()"


# ---------------------------------------------------------------------------
# Mixed-backend: arithmetic across backends must raise ValueError
# ---------------------------------------------------------------------------


class _MixedBackendClaim(Claim[None]):
    def __init__(self, label: str, fn: Any, a_backend: Any, b_backend: Any) -> None:
        self._label = label
        self._fn = fn
        self._a_backend = a_backend
        self._b_backend = b_backend

    @property
    def description(self) -> str:
        return (
            f"mixed_backend/{self._label}/"
            f"{type(self._a_backend).__name__}+{type(self._b_backend).__name__}"
        )

    def check(self, _calibration: None) -> None:
        a = Tensor([1.0, 2.0, 3.0], backend=self._a_backend)
        b = Tensor([1.0, 2.0, 3.0], backend=self._b_backend)
        with pytest.raises(ValueError, match="mix backends"):
            self._fn(a, b)


# ---------------------------------------------------------------------------
# Factory: Tensor.zeros and Tensor.eye respect the backend kwarg
# ---------------------------------------------------------------------------


class _FactoryClaim(Claim[None]):
    def __init__(self, backend: Any, name: str, fn: Any) -> None:
        self._backend = backend
        self._name = name
        self._fn = fn

    @property
    def description(self) -> str:
        return f"factory/{self._name}/{type(self._backend).__name__}"

    def check(self, _calibration: None) -> None:
        t = self._fn(self._backend)
        assert t.backend is self._backend, (
            f"{self.description}: expected backend {type(self._backend).__name__}, "
            f"got {type(t.backend).__name__}"
        )


# ---------------------------------------------------------------------------
# Slice read/write: all non-Python backends must agree with PythonBackend
# ---------------------------------------------------------------------------


class _SliceClaim(Claim[None]):
    """Claim: a slice read or write gives the same result as PythonBackend."""

    def __init__(self, label: str, fn: Any, backend: Any = None) -> None:
        self._label = label
        self._fn = fn
        self._backend = _NP if backend is None else backend

    @property
    def description(self) -> str:
        return f"slice/{self._label}/{type(self._backend).__name__}"

    def check(self, _calibration: None) -> None:
        py_result = self._fn(_PY)
        test_result = self._fn(self._backend)
        assert _approx_equal(py_result.to_list(), test_result.to_list()), (
            f"{self.description}: PythonBackend={py_result.to_list()!r} "
            f"!= {type(self._backend).__name__}={test_result.to_list()!r}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_equal(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _approx_equal(x, y, tol) for x, y in zip(a, b, strict=False)
        )
    return abs(float(a) - float(b)) <= tol


def _mk_vec(b: Any) -> Tensor:
    return Tensor([1.0, 2.0, 3.0], backend=b)


def _mk_mat(b: Any) -> Tensor:
    return Tensor([[1.0, 2.0], [3.0, 4.0]], backend=b)


def _mk_mat3(b: Any) -> Tensor:
    return Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], backend=b)


def _rank1_slice_write(b: Any) -> Tensor:
    t = Tensor([1.0, 2.0, 3.0, 4.0], backend=b)
    return t.set(slice(1, 3), Tensor([9.0, 9.0], backend=b))


def _rank2_col_slice_write(b: Any) -> Tensor:
    t = tensor.copy(_mk_mat3(b))
    return t.set((slice(1, 3), 0), Tensor([9.0, 9.0], backend=b))


def _rank2_submatrix_write(b: Any) -> Tensor:
    t = tensor.copy(_mk_mat3(b))
    return t.set(
        (slice(0, 2), slice(0, 2)), Tensor([[9.0, 8.0], [7.0, 6.0]], backend=b)
    )


# ---------------------------------------------------------------------------
# Case tables (shared across backends)
# ---------------------------------------------------------------------------

_ROUNDTRIP_CASES = [
    ("rank0", 3.14),
    ("rank1", [1.0, 2.0, 3.0]),
    ("rank2_2x2", [[1.0, 2.0], [3.0, 4.0]]),
    ("rank2_3x2", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
]

_ARITHMETIC_CASES = [
    ("add_vecs", lambda b: _mk_vec(b) + _mk_vec(b)),
    ("sub_vecs", lambda b: _mk_vec(b) - Tensor([0.5, 1.0, 1.5], backend=b)),
    ("neg_vec", lambda b: -_mk_vec(b)),
    ("mul_scalar", lambda b: _mk_vec(b) * 2.5),
    ("rmul_scalar", lambda b: 2.5 * _mk_vec(b)),
    ("div_scalar", lambda b: _mk_vec(b) / 4.0),
    ("mul_elem", lambda b: _mk_vec(b) * _mk_vec(b)),
    ("div_elem", lambda b: _mk_vec(b) / Tensor([2.0, 4.0, 1.0], backend=b)),
    ("dot", lambda b: _mk_vec(b) @ _mk_vec(b)),
    ("matvec", lambda b: _mk_mat(b) @ Tensor([1.0, 2.0], backend=b)),
    ("matmul", lambda b: _mk_mat(b) @ _mk_mat(b)),
    ("vecmat", lambda b: Tensor([1.0, 2.0], backend=b) @ _mk_mat(b)),
    ("einsum_ij_jk", lambda b: einsum("ij,jk->ik", _mk_mat(b), _mk_mat(b))),
    ("einsum_trace", lambda b: einsum("ii->", _mk_mat(b))),
    ("norm_vec", lambda b: Tensor([tensor.norm(_mk_vec(b)).get()], backend=b)),
    ("diag", lambda b: tensor.diag(_mk_mat(b))),
    ("zeros_factory", lambda b: Tensor.zeros(2, 3, backend=b)),
    ("eye_factory", lambda b: Tensor.eye(3, backend=b)),
    ("abs_vec", lambda b: tensor.abs(Tensor([-1.0, 2.0, -3.0], backend=b))),
    ("abs_mat", lambda b: tensor.abs(Tensor([[-1.0, 2.0], [3.0, -4.0]], backend=b))),
    ("max_vec", lambda b: tensor.max(Tensor([1.0, 5.0, 3.0], backend=b))),
    (
        "element_rank2",
        lambda b: tensor.element(Tensor([[1.0, 2.0], [3.0, 4.0]], backend=b), 1, 0),
    ),
    (
        "take_permute",
        lambda b: tensor.take(
            Tensor([10.0, 20.0, 30.0], backend=b), Tensor([2, 0, 1], backend=b)
        ),
    ),
    ("rdiv_scalar", lambda b: 6.0 / Tensor([1.0, 2.0, 3.0], backend=b)),
    (
        "lt_vecs",
        lambda b: Tensor([1.0, 3.0, 2.0], backend=b)
        < Tensor([2.0, 2.0, 2.0], backend=b),
    ),
    (
        "gt_scalar",
        lambda b: Tensor([1.0, 3.0, 2.0], backend=b) > 2.0,
    ),
    (
        "where_array_cond",
        lambda b: where(
            Tensor([True, False, True], backend=b),
            Tensor([1.0, 2.0, 3.0], backend=b),
            Tensor([4.0, 5.0, 6.0], backend=b),
        ),
    ),
    (
        "where_scalar_cond_true",
        lambda b: where(Tensor(True, backend=b), Tensor([1.0, 2.0], backend=b), 0.0),
    ),
    ("arange_4", lambda b: arange(4, backend=b)),
    ("argmax_val", lambda b: tensor.argmax(Tensor([1.0, 3.0, 2.0], backend=b))),
    (
        "dynamic_index",
        lambda b: Tensor([10.0, 20.0, 30.0], backend=b)[
            tensor.argmax(Tensor([1.0, 3.0, 2.0], backend=b))
        ],
    ),
]

_SLICE_READ_CASES = [
    ("rank1_read", lambda b: Tensor([1.0, 2.0, 3.0, 4.0], backend=b)[1:3]),
    ("rank1_read_from_start", lambda b: Tensor([1.0, 2.0, 3.0], backend=b)[:2]),
    ("rank1_read_to_end", lambda b: Tensor([1.0, 2.0, 3.0], backend=b)[1:]),
    ("rank2_row_read", lambda b: _mk_mat3(b)[1, :]),
    ("rank2_row_partial", lambda b: _mk_mat3(b)[0, 1:]),
    ("rank2_col_read", lambda b: _mk_mat3(b)[:, 1]),
    ("rank2_col_partial", lambda b: _mk_mat3(b)[1:, 0]),
    ("rank2_submatrix", lambda b: _mk_mat3(b)[1:, 1:]),
    ("rank1_int_index", lambda b: Tensor([1.0, 2.0, 3.0, 4.0], backend=b)[2]),
]

_SLICE_WRITE_CASES = [
    ("rank1_write", _rank1_slice_write),
    ("rank2_col_write", _rank2_col_slice_write),
    ("rank2_submatrix_write", _rank2_submatrix_write),
]

_CONVERSION_PAIRS = [
    (_PY, _NP),
    (_NP, _PY),
    (_PY, _JAX),
    (_JAX, _PY),
    (_NP, _JAX),
    (_JAX, _NP),
]

_CONVERSION_DATA = [
    ("rank1", [1.0, 2.0, 3.0]),
    ("rank2", [[1.0, 2.0], [3.0, 4.0]]),
]

_MIXED_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul_elem", lambda a, b: a * b),
    ("matmul", lambda a, b: a @ b),
]

# ---------------------------------------------------------------------------
# Correctness Registry
# ---------------------------------------------------------------------------

_CORRECTNESS_CLAIMS: list[Claim[None]] = [
    # Round-trips for all backends
    *[
        _RoundtripClaim(b, label, data)
        for b in (_PY, _NP, _JAX)
        for label, data in _ROUNDTRIP_CASES
    ],
    # Arithmetic: each fn tested against PY reference on NP and JAX
    *[
        _ArithmeticClaim(label, fn, b)
        for b in _NON_PY_BACKENDS
        for label, fn in _ARITHMETIC_CASES
    ],
    # Conversion round-trips
    *[
        _ConversionClaim(src, dst, label, data)
        for src, dst in _CONVERSION_PAIRS
        for label, data in _CONVERSION_DATA
    ],
    # Slice reads: each result must match PythonBackend
    *[
        _SliceClaim(label, fn, b)
        for b in _NON_PY_BACKENDS
        for label, fn in _SLICE_READ_CASES
    ],
    # Slice writes: modify a copy and return it; must match PythonBackend
    *[
        _SliceClaim(label, fn, b)
        for b in _NON_PY_BACKENDS
        for label, fn in _SLICE_WRITE_CASES
    ],
    # Mixed-backend must raise: all pairs of distinct backends
    *[
        _MixedBackendClaim(label, fn, a_b, b_b)
        for label, fn in _MIXED_OPS
        for a_b, b_b in [(_PY, _NP), (_PY, _JAX), (_NP, _JAX)]
    ],
    # Factories respect backend kwarg
    *[
        _FactoryClaim(b, name, fn)
        for b in (_PY, _NP, _JAX)
        for name, fn in [
            ("zeros", lambda b: Tensor.zeros(3, backend=b)),
            ("eye", lambda b: Tensor.eye(3, backend=b)),
        ]
    ],
]


@pytest.mark.parametrize(
    "claim", _CORRECTNESS_CLAIMS, ids=[c.description for c in _CORRECTNESS_CLAIMS]
)
def test_correctness(claim: Claim[None]) -> None:
    claim.check(None)


# Regressions larger than this multiple of the roofline prediction fail.
EFFICIENCY_FACTOR = 8

# NumpyBackend Tensor must stay within this multiple of raw NumPy throughput.
NUMPY_PARITY_FACTOR = 2

# Number of trials; the minimum time across trials is used to eliminate
# OS scheduling noise while still catching algorithmic slowdowns.
_TRIALS = 20


@dataclass(frozen=True)
class _TensorPerformanceCalibration:
    fma_rate: float
    device_calibration: DeviceCalibration


@pytest.fixture(scope="module")
def tensor_performance_calibration(
    fma_rate: float, device_calibration: DeviceCalibration
) -> _TensorPerformanceCalibration:
    return _TensorPerformanceCalibration(fma_rate, device_calibration)


# ---------------------------------------------------------------------------
# Performance claim classes
# ---------------------------------------------------------------------------


class _MatvecPerfClaim(Claim[_TensorPerformanceCalibration]):
    """Claim: N×N @ N matvec runs within EFFICIENCY_FACTOR of the FMA roofline.

    Expected FMAs: 2N² (N rows × N multiply-adds).
    """

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def description(self) -> str:
        return f"matvec/N={self._n}"

    def check(self, calibration: _TensorPerformanceCalibration) -> None:
        n = self._n
        a = Tensor([[float(i + j) for j in range(n)] for i in range(n)])
        x = Tensor([float(i) for i in range(n)])

        best_elapsed = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            _ = a @ x
            best_elapsed = min(best_elapsed, time.perf_counter() - t0)

        expected = 2 * n**2 / calibration.fma_rate
        assert best_elapsed <= EFFICIENCY_FACTOR * expected, (
            f"matvec N={n}: {best_elapsed * 1e6:.1f}µs actual, "
            f"{expected * 1e6:.1f}µs roofline, "
            f"{best_elapsed / expected:.1f}× > {EFFICIENCY_FACTOR}× limit"
        )


class _MatmulPerfClaim(Claim[_TensorPerformanceCalibration]):
    """Claim: N×N @ N×N matmul runs within EFFICIENCY_FACTOR of the FMA roofline.

    Expected FMAs: 2N³ (N² output elements × N multiply-adds each).
    """

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def description(self) -> str:
        return f"matmul/N={self._n}"

    def check(self, calibration: _TensorPerformanceCalibration) -> None:
        n = self._n
        a = Tensor([[float(i + j) for j in range(n)] for i in range(n)])
        b = Tensor([[float(i * j + 1) for j in range(n)] for i in range(n)])

        best_elapsed = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            _ = a @ b
            best_elapsed = min(best_elapsed, time.perf_counter() - t0)

        expected = 2 * n**3 / calibration.fma_rate
        assert best_elapsed <= EFFICIENCY_FACTOR * expected, (
            f"matmul N={n}: {best_elapsed * 1e6:.1f}µs actual, "
            f"{expected * 1e6:.1f}µs roofline, "
            f"{best_elapsed / expected:.1f}× > {EFFICIENCY_FACTOR}× limit"
        )


class _DotPerfClaim(Claim[_TensorPerformanceCalibration]):
    """Claim: N @ N dot product runs within EFFICIENCY_FACTOR of the FMA roofline.

    Expected FMAs: 2N (N multiplies + N-1 adds ≈ 2N).
    """

    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def description(self) -> str:
        return f"dot/N={self._n}"

    def check(self, calibration: _TensorPerformanceCalibration) -> None:
        n = self._n
        a = Tensor([float(i) for i in range(n)])
        b = Tensor([float(i) * 0.5 for i in range(n)])

        best_elapsed = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            _ = a @ b
            best_elapsed = min(best_elapsed, time.perf_counter() - t0)

        expected = 2 * n / calibration.fma_rate
        assert best_elapsed <= EFFICIENCY_FACTOR * expected, (
            f"dot N={n}: {best_elapsed * 1e6:.1f}µs actual, "
            f"{expected * 1e6:.1f}µs roofline, "
            f"{best_elapsed / expected:.1f}× > {EFFICIENCY_FACTOR}× limit"
        )


class _NumpyParityPerfClaim(Claim[_TensorPerformanceCalibration]):
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

    def check(self, calibration: _TensorPerformanceCalibration) -> None:
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


class _BackendSpeedupClaim(Claim[_TensorPerformanceCalibration]):
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

    def check(self, calibration: _TensorPerformanceCalibration) -> None:
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
# Device performance model constants
# ---------------------------------------------------------------------------

# Post-warmup backend Tensor ops must stay within this multiple of the
# JIT-compiled roofline.  The gap reflects eager dispatch overhead; the bound
# catches catastrophic regressions (e.g. accidental CPU fallback on GPU).
_DEVICE_EFFICIENCY_FACTOR = 10

# GPU JIT roofline must be at least this many times the CPU JIT roofline.
_DEVICE_GPU_CPU_MIN_SPEEDUP = 2

# Warmup calls to issue before timed trials so the dispatch cache is warm.
_DEVICE_WARMUP = 3


# ---------------------------------------------------------------------------
# Device claim classes
# ---------------------------------------------------------------------------


class _DeviceCpuPerfClaim(Claim[_TensorPerformanceCalibration]):
    """Backend CPU op ≤ _DEVICE_EFFICIENCY_FACTOR × CPU JIT roofline (post-warmup)."""

    def __init__(self, op: str, n: int) -> None:
        self._op = op
        self._n = n

    @property
    def description(self) -> str:
        return f"device_cpu/{self._op}/N={self._n}"

    def check(self, calibration: _TensorPerformanceCalibration) -> None:
        device_calibration = calibration.device_calibration
        backend = device_calibration.cpu_backend
        n = self._n
        if self._op == "matmul":
            a = Tensor(
                [[float(i + j) for j in range(n)] for i in range(n)], backend=backend
            )
            b = Tensor(
                [[float(i * j + 1) for j in range(n)] for i in range(n)],
                backend=backend,
            )
            expected_fmas = 2 * n**3
        else:
            a = Tensor(
                [[float(i + j) for j in range(n)] for i in range(n)], backend=backend
            )
            b = Tensor([float(i) for i in range(n)], backend=backend)
            expected_fmas = 2 * n**2
        for _ in range(_DEVICE_WARMUP):
            r = a @ b
            r.sync()
        best_elapsed = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            r = a @ b
            r.sync()
            best_elapsed = min(best_elapsed, time.perf_counter() - t0)
        expected = expected_fmas / device_calibration.cpu_fma_rate
        assert best_elapsed <= _DEVICE_EFFICIENCY_FACTOR * expected, (
            f"{self.description}: "
            f"{best_elapsed * 1e6:.1f}µs actual, "
            f"{expected * 1e6:.1f}µs roofline, "
            f"{best_elapsed / expected:.1f}× > {_DEVICE_EFFICIENCY_FACTOR}× limit"
        )


class _DeviceGpuPerfClaim(Claim[_TensorPerformanceCalibration]):
    """Backend GPU op ≤ _DEVICE_EFFICIENCY_FACTOR × GPU JIT roofline (post-warmup).

    Skipped automatically when device_calibration.gpu_fma_rate is None.
    """

    def __init__(self, op: str, n: int) -> None:
        self._op = op
        self._n = n

    @property
    def description(self) -> str:
        return f"device_gpu/{self._op}/N={self._n}"

    def check(self, calibration: _TensorPerformanceCalibration) -> None:
        device_calibration = calibration.device_calibration
        if device_calibration.gpu_fma_rate is None:
            pytest.skip("no GPU device available")
        backend = device_calibration.gpu_backend
        n = self._n
        if self._op == "matmul":
            a = Tensor(
                [[float(i + j) for j in range(n)] for i in range(n)],
                backend=backend,
            )
            b = Tensor(
                [[float(i * j + 1) for j in range(n)] for i in range(n)],
                backend=backend,
            )
            expected_fmas = 2 * n**3
        else:
            a = Tensor(
                [[float(i + j) for j in range(n)] for i in range(n)],
                backend=backend,
            )
            b = Tensor([float(i) for i in range(n)], backend=backend)
            expected_fmas = 2 * n**2
        for _ in range(_DEVICE_WARMUP):
            r = a @ b
            r.sync()
        best_elapsed = float("inf")
        for _ in range(_TRIALS):
            t0 = time.perf_counter()
            r = a @ b
            r.sync()
            best_elapsed = min(best_elapsed, time.perf_counter() - t0)
        expected = expected_fmas / device_calibration.gpu_fma_rate
        assert best_elapsed <= _DEVICE_EFFICIENCY_FACTOR * expected, (
            f"{self.description}: "
            f"{best_elapsed * 1e6:.1f}µs actual, "
            f"{expected * 1e6:.1f}µs roofline, "
            f"{best_elapsed / expected:.1f}× > {_DEVICE_EFFICIENCY_FACTOR}× limit"
        )


class _DeviceGpuVsCpuRooflineClaim(Claim[_TensorPerformanceCalibration]):
    """GPU JIT roofline ≥ min_speedup × CPU JIT roofline.

    Catches miscalibration (e.g. GPU measurement accidentally ran on CPU)
    and GPU configurations where the backend falls back to a CPU-speed device.
    Skipped when no GPU device is available.
    """

    def __init__(self, min_speedup: int) -> None:
        self._min_speedup = min_speedup

    @property
    def description(self) -> str:
        return f"device_gpu_vs_cpu_roofline/{self._min_speedup}x"

    def check(self, calibration: _TensorPerformanceCalibration) -> None:
        device_calibration = calibration.device_calibration
        if device_calibration.gpu_fma_rate is None:
            pytest.skip("no GPU device available")
        speedup = device_calibration.gpu_fma_rate / device_calibration.cpu_fma_rate
        assert speedup >= self._min_speedup, (
            f"GPU roofline {device_calibration.gpu_fma_rate:.2e} FMAs/s is only "
            f"{speedup:.1f}× CPU roofline {device_calibration.cpu_fma_rate:.2e} FMAs/s "
            f"(required ≥ {self._min_speedup}×)"
        )


# ---------------------------------------------------------------------------
# Performance Registries
# ---------------------------------------------------------------------------

_PERF_CLAIMS: list[Claim[_TensorPerformanceCalibration]] = [
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
    # Backend CPU vs CPU JIT roofline (dot-product-calibrated, dispatch-limited)
    *[_DeviceCpuPerfClaim("matmul", n) for n in [128, 256]],
    *[_DeviceCpuPerfClaim("matvec", n) for n in [128, 256]],
    # Backend GPU matmul vs GPU JIT roofline (matmul-calibrated, compute-bound)
    # Matvec is omitted: at all testable N, GPU kernel launch overhead (~1 ms) swamps
    # the tiny matvec compute, making the ratio meaninglessly large.
    *[_DeviceGpuPerfClaim("matmul", n) for n in [256, 512, 1024]],
    # Cross-device sanity check: GPU compute roofline ≥ 2× CPU dispatch-limited baseline
    _DeviceGpuVsCpuRooflineClaim(min_speedup=_DEVICE_GPU_CPU_MIN_SPEEDUP),
]


@pytest.mark.parametrize(
    "claim", _PERF_CLAIMS, ids=[c.description for c in _PERF_CLAIMS]
)
def test_performance(
    claim: Claim[_TensorPerformanceCalibration],
    tensor_performance_calibration: _TensorPerformanceCalibration,
) -> None:
    claim.check(tensor_performance_calibration)
