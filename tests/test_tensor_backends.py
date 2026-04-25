"""Correctness tests for Tensor backends and the pluggable backend system.

Each claim encodes one correctness property of the Tensor / Backend interface.
Adding a new claim requires only appending to _CLAIMS; the single parametric
test covers all entries.

Claim types:
  _RoundtripClaim(backend)       — to_native → from_native preserves values
  _ArithmeticClaim(backend, op)  — arithmetic result matches PythonBackend reference
  _ConversionClaim(src, dst)     — .to() preserves shape and element values
  _MixedBackendClaim(op)         — mixing backends raises ValueError
  _FactoryClaim(backend, name)   — Tensor.zeros / Tensor.eye use the given backend
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pytest

from cosmic_foundry.computation.backends import NumpyBackend, PythonBackend
from cosmic_foundry.computation.tensor import Tensor, einsum

_PY = PythonBackend()
_NP = NumpyBackend()


class _TensorClaim(ABC):
    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self) -> None: ...


# ---------------------------------------------------------------------------
# Round-trip: to_native → from_native leaves values unchanged
# ---------------------------------------------------------------------------


class _RoundtripClaim(_TensorClaim):
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
# Arithmetic: NumpyBackend result matches PythonBackend reference
# ---------------------------------------------------------------------------


class _ArithmeticClaim(_TensorClaim):
    def __init__(self, label: str, fn: Any) -> None:
        self._label = label
        self._fn = fn

    @property
    def description(self) -> str:
        return f"arithmetic/{self._label}"

    def check(self) -> None:
        py_result = self._fn(_PY)
        np_result = self._fn(_NP)
        assert _approx_equal(py_result.to_list(), np_result.to_list()), (
            f"{self.description}: PythonBackend={py_result.to_list()!r} "
            f"NumpyBackend={np_result.to_list()!r}"
        )


# ---------------------------------------------------------------------------
# Conversion: .to() preserves shape and values across backends
# ---------------------------------------------------------------------------


class _ConversionClaim(_TensorClaim):
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


class _MixedBackendClaim(_TensorClaim):
    def __init__(self, label: str, fn: Any) -> None:
        self._label = label
        self._fn = fn

    @property
    def description(self) -> str:
        return f"mixed_backend/{self._label}"

    def check(self) -> None:
        py_t = Tensor([1.0, 2.0, 3.0], backend=_PY)
        np_t = Tensor([1.0, 2.0, 3.0], backend=_NP)
        with pytest.raises(ValueError, match="mix backends"):
            self._fn(py_t, np_t)


# ---------------------------------------------------------------------------
# Factory: Tensor.zeros and Tensor.eye respect the backend kwarg
# ---------------------------------------------------------------------------


class _FactoryClaim(_TensorClaim):
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
# Slice read/write: PythonBackend and NumpyBackend must agree
# ---------------------------------------------------------------------------


class _SliceClaim(_TensorClaim):
    """Claim: a slice read or write gives the same result on both backends."""

    def __init__(self, label: str, fn: Any) -> None:
        self._label = label
        self._fn = fn

    @property
    def description(self) -> str:
        return f"slice/{self._label}"

    def check(self) -> None:
        py_result = self._fn(_PY)
        np_result = self._fn(_NP)
        assert _approx_equal(py_result.to_list(), np_result.to_list()), (
            f"{self.description}: PythonBackend={py_result.to_list()!r} "
            f"!= NumpyBackend={np_result.to_list()!r}"
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
    t[1:3] = Tensor([9.0, 9.0], backend=b)
    return t


def _rank2_col_slice_write(b: Any) -> Tensor:
    t = _mk_mat3(b).copy()
    t[1:3, 0] = Tensor([9.0, 9.0], backend=b)
    return t


def _rank2_submatrix_write(b: Any) -> Tensor:
    t = _mk_mat3(b).copy()
    t[0:2, 0:2] = Tensor([[9.0, 8.0], [7.0, 6.0]], backend=b)
    return t


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CLAIMS: list[_TensorClaim] = [
    # Round-trips
    *[
        _RoundtripClaim(b, label, data)
        for b in (_PY, _NP)
        for label, data in [
            ("rank0", 3.14),
            ("rank1", [1.0, 2.0, 3.0]),
            ("rank2_2x2", [[1.0, 2.0], [3.0, 4.0]]),
            ("rank2_3x2", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        ]
    ],
    # Arithmetic: result of each op matches between backends
    _ArithmeticClaim(
        "add_vecs",
        lambda b: _mk_vec(b) + _mk_vec(b),
    ),
    _ArithmeticClaim(
        "sub_vecs",
        lambda b: _mk_vec(b) - Tensor([0.5, 1.0, 1.5], backend=b),
    ),
    _ArithmeticClaim(
        "neg_vec",
        lambda b: -_mk_vec(b),
    ),
    _ArithmeticClaim(
        "mul_scalar",
        lambda b: _mk_vec(b) * 2.5,
    ),
    _ArithmeticClaim(
        "rmul_scalar",
        lambda b: 2.5 * _mk_vec(b),
    ),
    _ArithmeticClaim(
        "div_scalar",
        lambda b: _mk_vec(b) / 4.0,
    ),
    _ArithmeticClaim(
        "mul_elem",
        lambda b: _mk_vec(b) * _mk_vec(b),
    ),
    _ArithmeticClaim(
        "div_elem",
        lambda b: _mk_vec(b) / Tensor([2.0, 4.0, 1.0], backend=b),
    ),
    _ArithmeticClaim(
        "dot",
        lambda b: _mk_vec(b) @ _mk_vec(b),
    ),
    _ArithmeticClaim(
        "matvec",
        lambda b: _mk_mat(b) @ Tensor([1.0, 2.0], backend=b),
    ),
    _ArithmeticClaim(
        "matmul",
        lambda b: _mk_mat(b) @ _mk_mat(b),
    ),
    _ArithmeticClaim(
        "vecmat",
        lambda b: Tensor([1.0, 2.0], backend=b) @ _mk_mat(b),
    ),
    _ArithmeticClaim(
        "einsum_ij_jk",
        lambda b: einsum("ij,jk->ik", _mk_mat(b), _mk_mat(b)),
    ),
    _ArithmeticClaim(
        "einsum_trace",
        lambda b: einsum("ii->", _mk_mat(b)),
    ),
    _ArithmeticClaim(
        "norm_vec",
        lambda b: Tensor(
            [_mk_vec(b).norm()],
            backend=b,
        ),
    ),
    _ArithmeticClaim(
        "diag",
        lambda b: _mk_mat(b).diag(),
    ),
    _ArithmeticClaim(
        "zeros_factory",
        lambda b: Tensor.zeros(2, 3, backend=b),
    ),
    _ArithmeticClaim(
        "eye_factory",
        lambda b: Tensor.eye(3, backend=b),
    ),
    # Conversion round-trips
    *[
        _ConversionClaim(src, dst, label, data)
        for src, dst in [(_PY, _NP), (_NP, _PY)]
        for label, data in [
            ("rank1", [1.0, 2.0, 3.0]),
            ("rank2", [[1.0, 2.0], [3.0, 4.0]]),
        ]
    ],
    # Slice reads: each result must match between PythonBackend and NumpyBackend
    _SliceClaim("rank1_read", lambda b: Tensor([1.0, 2.0, 3.0, 4.0], backend=b)[1:3]),
    _SliceClaim(
        "rank1_read_from_start", lambda b: Tensor([1.0, 2.0, 3.0], backend=b)[:2]
    ),
    _SliceClaim("rank1_read_to_end", lambda b: Tensor([1.0, 2.0, 3.0], backend=b)[1:]),
    _SliceClaim("rank2_row_read", lambda b: _mk_mat3(b)[1, :]),
    _SliceClaim("rank2_row_partial", lambda b: _mk_mat3(b)[0, 1:]),
    _SliceClaim("rank2_col_read", lambda b: _mk_mat3(b)[:, 1]),
    _SliceClaim("rank2_col_partial", lambda b: _mk_mat3(b)[1:, 0]),
    _SliceClaim("rank2_submatrix", lambda b: _mk_mat3(b)[1:, 1:]),
    # Slice writes: modify a copy and return it; must match between backends
    _SliceClaim("rank1_write", _rank1_slice_write),
    _SliceClaim("rank2_col_write", _rank2_col_slice_write),
    _SliceClaim("rank2_submatrix_write", _rank2_submatrix_write),
    # Mixed-backend must raise
    _MixedBackendClaim("add", lambda a, b: a + b),
    _MixedBackendClaim("sub", lambda a, b: a - b),
    _MixedBackendClaim("mul_elem", lambda a, b: a * b),
    _MixedBackendClaim("matmul", lambda a, b: a @ b),
    # Factories respect backend kwarg
    _FactoryClaim(_PY, "zeros", lambda b: Tensor.zeros(3, backend=b)),
    _FactoryClaim(_NP, "zeros", lambda b: Tensor.zeros(3, backend=b)),
    _FactoryClaim(_PY, "eye", lambda b: Tensor.eye(3, backend=b)),
    _FactoryClaim(_NP, "eye", lambda b: Tensor.eye(3, backend=b)),
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_tensor_backend(claim: _TensorClaim) -> None:
    claim.check()
