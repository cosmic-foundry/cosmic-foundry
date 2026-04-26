"""Tensor correctness claims for all backends.

Each claim encodes one correctness property of the Tensor / Backend interface.
Adding a new claim requires only appending to _CLAIMS; the single parametric
test covers all entries.

Claim types:
  _RoundtripClaim(backend)       — to_native → from_native preserves values
  _ArithmeticClaim(backend, op)  — arithmetic result matches PythonBackend reference
  _ConversionClaim(src, dst)     — .to() preserves shape and element values
  _MixedBackendClaim(op)         — mixing backends raises ValueError
  _FactoryClaim(backend, name)   — Tensor.zeros / Tensor.eye use the given backend
  _SliceClaim(backend, op)       — slice read/write matches PythonBackend reference
"""

from __future__ import annotations

from typing import Any

import pytest

from cosmic_foundry.computation.backends import JaxBackend, NumpyBackend, PythonBackend
from cosmic_foundry.computation.tensor import Tensor, arange, einsum, where
from tests.claims import Claim

_PY = PythonBackend()
_NP = NumpyBackend()
_JAX = JaxBackend()

_NON_PY_BACKENDS = (_NP, _JAX)


# ---------------------------------------------------------------------------
# Round-trip: to_native → from_native leaves values unchanged
# ---------------------------------------------------------------------------


class _RoundtripClaim(Claim):
    def __init__(self, backend: Any, label: str, data: Any) -> None:
        self._backend = backend
        self._label = label
        self._data = data

    @property
    def description(self) -> str:
        return f"roundtrip/{self._label}/{type(self._backend).__name__}"

    def check(self) -> None:
        t = Tensor(self._data, backend=self._backend)
        recovered = t.to_list()
        ref = Tensor(self._data, backend=_PY).to_list()
        assert _approx_equal(
            recovered, ref
        ), f"{self.description}: got {recovered!r}, expected {ref!r}"


# ---------------------------------------------------------------------------
# Arithmetic: backend result matches PythonBackend reference
# ---------------------------------------------------------------------------


class _ArithmeticClaim(Claim):
    def __init__(self, label: str, fn: Any, backend: Any = None) -> None:
        self._label = label
        self._fn = fn
        self._backend = _NP if backend is None else backend

    @property
    def description(self) -> str:
        return f"arithmetic/{self._label}/{type(self._backend).__name__}"

    def check(self) -> None:
        py_result = self._fn(_PY)
        test_result = self._fn(self._backend)
        assert _approx_equal(py_result.to_list(), test_result.to_list()), (
            f"{self.description}: PythonBackend={py_result.to_list()!r} "
            f"{type(self._backend).__name__}={test_result.to_list()!r}"
        )


# ---------------------------------------------------------------------------
# Conversion: .to() preserves shape and values across backends
# ---------------------------------------------------------------------------


class _ConversionClaim(Claim):
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

    def check(self) -> None:
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


class _MixedBackendClaim(Claim):
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

    def check(self) -> None:
        a = Tensor([1.0, 2.0, 3.0], backend=self._a_backend)
        b = Tensor([1.0, 2.0, 3.0], backend=self._b_backend)
        with pytest.raises(ValueError, match="mix backends"):
            self._fn(a, b)


# ---------------------------------------------------------------------------
# Factory: Tensor.zeros and Tensor.eye respect the backend kwarg
# ---------------------------------------------------------------------------


class _FactoryClaim(Claim):
    def __init__(self, backend: Any, name: str, fn: Any) -> None:
        self._backend = backend
        self._name = name
        self._fn = fn

    @property
    def description(self) -> str:
        return f"factory/{self._name}/{type(self._backend).__name__}"

    def check(self) -> None:
        t = self._fn(self._backend)
        assert t.backend is self._backend, (
            f"{self.description}: expected backend {type(self._backend).__name__}, "
            f"got {type(t.backend).__name__}"
        )


# ---------------------------------------------------------------------------
# Slice read/write: all non-Python backends must agree with PythonBackend
# ---------------------------------------------------------------------------


class _SliceClaim(Claim):
    """Claim: a slice read or write gives the same result as PythonBackend."""

    def __init__(self, label: str, fn: Any, backend: Any = None) -> None:
        self._label = label
        self._fn = fn
        self._backend = _NP if backend is None else backend

    @property
    def description(self) -> str:
        return f"slice/{self._label}/{type(self._backend).__name__}"

    def check(self) -> None:
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
    t = _mk_mat3(b).copy()
    return t.set((slice(1, 3), 0), Tensor([9.0, 9.0], backend=b))


def _rank2_submatrix_write(b: Any) -> Tensor:
    t = _mk_mat3(b).copy()
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
    ("norm_vec", lambda b: Tensor([_mk_vec(b).norm().get()], backend=b)),
    ("diag", lambda b: _mk_mat(b).diag()),
    ("zeros_factory", lambda b: Tensor.zeros(2, 3, backend=b)),
    ("eye_factory", lambda b: Tensor.eye(3, backend=b)),
    ("abs_vec", lambda b: Tensor([-1.0, 2.0, -3.0], backend=b).abs()),
    ("abs_mat", lambda b: Tensor([[-1.0, 2.0], [3.0, -4.0]], backend=b).abs()),
    ("max_vec", lambda b: Tensor([1.0, 5.0, 3.0], backend=b).max()),
    (
        "element_rank2",
        lambda b: Tensor([[1.0, 2.0], [3.0, 4.0]], backend=b).element(1, 0),
    ),
    (
        "take_permute",
        lambda b: Tensor([10.0, 20.0, 30.0], backend=b).take(
            Tensor([2, 0, 1], backend=b)
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
    ("argmax_val", lambda b: Tensor([1.0, 3.0, 2.0], backend=b).argmax()),
    (
        "dynamic_index",
        lambda b: Tensor([10.0, 20.0, 30.0], backend=b)[
            Tensor([1.0, 3.0, 2.0], backend=b).argmax()
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
# Registry
# ---------------------------------------------------------------------------

_CLAIMS: list[Claim] = [
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


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_tensor(claim: Claim) -> None:
    claim.check()
