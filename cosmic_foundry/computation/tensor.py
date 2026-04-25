"""Tensor: a pure-Python numeric array of arbitrary rank.

All arithmetic is implemented with Python lists and the math module.
No NumPy or JAX is used.  This is the reference implementation — kept
permanently for readability and cross-checking against higher-performance
backends.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Real(Protocol):
    """Protocol for scalar numeric types usable as Tensor elements.

    Any type satisfying these operations qualifies: Python float, int,
    numpy.float16/32/64, and JAX scalar types all conform.  The protocol
    covers exactly the operations Tensor performs on its elements.
    """

    def __neg__(self) -> Real: ...
    def __add__(self, other: Any) -> Real: ...
    def __radd__(self, other: Any) -> Real: ...
    def __sub__(self, other: Any) -> Real: ...
    def __rsub__(self, other: Any) -> Real: ...
    def __mul__(self, other: Any) -> Real: ...
    def __rmul__(self, other: Any) -> Real: ...
    def __truediv__(self, other: Any) -> Real: ...
    def __float__(self) -> float: ...


class Tensor:
    """A numeric array of arbitrary rank backed by nested Python lists.

    Leaf elements must satisfy the Real protocol.  Rank is inferred from
    the nesting depth of the input; all slices along any axis must have
    the same length (jagged arrays are not supported).

    Construction:
        Tensor(1.0)                        — rank-0 scalar
        Tensor([1.0, 2.0, 3.0])            — rank-1, shape (3,)
        Tensor([[1.0, 2.0], [3.0, 4.0]])   — rank-2, shape (2, 2)
        Tensor([[[...], ...], ...])         — rank-3+, arbitrary shape
        Tensor.zeros()                      — rank-0 zero
        Tensor.zeros(m, n, k)              — rank-3 zero array
        Tensor.eye(n)                       — n × n identity

    Indexing (rank ≥ 1 only):
        t[i]         — Real for rank-1, rank-(n-1) Tensor for rank-n
        t[i, j, ...]  — Real when all indices supplied, Tensor otherwise
        t[i] = v     — assign element (Real) or sub-tensor to position i
        t[i, j] = v  — assign element or sub-tensor at multi-index

    Arithmetic (element-wise, any rank):
        a + b, a - b, -a
        a * scalar, scalar * a, a / scalar
        float(t)  — extract value from rank-0 Tensor

    Tensor contraction:
        einsum(spec, a, b, ...)  — general Einstein summation
        a @ b  — einsum delegate: dot / vecmat / matvec / matmul

    Linear algebra (rank-2 only):
        t.diag()   — main diagonal → rank-1
        t.svd()    — (U, s, Vt); singular values descending

    Any rank:
        t.norm()    — Frobenius norm (|value| for rank-0) → float
        t.to_list() — underlying data; scalar for rank-0, nested list otherwise
    """

    def __init__(self, data: Any) -> None:
        if isinstance(data, Sequence) and not isinstance(data, str | bytes):
            # rank ≥ 1: recursively convert to nested lists.
            self._data: Any = _to_list(data)
            self._shape: tuple[int, ...] = _infer_shape(self._data)
        else:
            # rank-0 scalar.
            self._data = data
            self._shape = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __float__(self) -> float:
        """Extract the scalar value from a rank-0 Tensor."""
        if self._shape:
            raise TypeError(
                f"cannot convert rank-{len(self._shape)} Tensor to float; "
                "index to a scalar or call .norm() first"
            )
        return float(self._data)

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("rank-0 Tensor has no length")
        return self._shape[0]

    def __getitem__(self, idx: int | tuple[int, ...]) -> Any:
        if not self._shape:
            raise TypeError("rank-0 Tensor is not subscriptable")
        if isinstance(idx, tuple):
            result: Any = self
            for i in idx:
                if not isinstance(result, Tensor):
                    raise IndexError("too many indices for Tensor")
                result = result[i]
            return result
        if len(self._shape) == 1:
            return self._data[idx]
        return Tensor(self._data[idx])

    def __setitem__(self, idx: int | tuple[int, ...], value: Any) -> None:
        if not self._shape:
            raise TypeError("rank-0 Tensor is not subscriptable")
        v = value._data if isinstance(value, Tensor) else value
        if isinstance(idx, tuple):
            if not idx:
                raise IndexError("empty index tuple")
            data = self._data
            for i in idx[:-1]:
                data = data[i]
            data[idx[-1]] = v
        else:
            self._data[idx] = v

    def __iter__(self) -> Iterator[Any]:
        if not self._shape:
            raise TypeError("rank-0 Tensor is not iterable")
        for i in range(self._shape[0]):
            yield self[i]

    def to_list(self) -> Any:
        """Scalar for rank-0; deep-copied nested list for rank ≥ 1."""
        return _deep_copy(self._data)

    def __repr__(self) -> str:
        return f"Tensor({self._data!r})"

    def __neg__(self) -> Tensor:
        return Tensor(_map(self._data, lambda x: -x))

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor(_zip_map(self._data, other._data, lambda x, y: x + y))

    def __sub__(self, other: Tensor) -> Tensor:
        return Tensor(_zip_map(self._data, other._data, lambda x, y: x - y))

    def __mul__(self, scalar: float) -> Tensor:
        s = float(scalar)
        return Tensor(_map(self._data, lambda x: x * s))

    def __rmul__(self, scalar: float) -> Tensor:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Tensor:
        return self.__mul__(1.0 / float(scalar))

    def __matmul__(self, other: Tensor) -> Tensor:
        r1, r2 = len(self._shape), len(other._shape)
        if r1 == 0 or r2 == 0:
            raise ValueError(f"unsupported matmul: {self._shape} @ {other._shape}")
        return einsum(_matmul_spec(r1, r2), self, other)

    def diag(self) -> Tensor:
        """Main diagonal of a rank-2 Tensor → rank-1."""
        if len(self._shape) != 2:
            raise ValueError(f"diag requires rank-2 Tensor, got shape {self._shape}")
        n = min(self._shape)
        return Tensor([self._data[i][i] for i in range(n)])

    def norm(self) -> float:
        """Frobenius norm (absolute value for rank-0)."""
        if not self._shape:
            return abs(float(self._data))
        return math.sqrt(sum(x * x for x in _flatten(self._data)))

    def svd(self) -> tuple[Tensor, Tensor, Tensor]:
        """One-sided Jacobi SVD: returns (U, s, Vt) with singular values descending.

        Applies cyclic Givens rotations to pairs of columns of A until all
        columns are mutually orthogonal (one-sided algorithm; no explicit
        formation of AᵀA).  Convergence is guaranteed for any real matrix.

        Returns
        -------
        U   : Tensor of shape (m, n) — left singular vectors as columns
        s   : Tensor of shape (n,)   — singular values, descending order
        Vt  : Tensor of shape (n, n) — right singular vectors as rows
        """
        if len(self._shape) != 2:
            raise ValueError(f"svd requires rank-2 Tensor, got shape {self._shape}")
        m, n = self._shape

        # B[j] = column j of A (length m); updated in-place by Givens rotations.
        B = [[self._data[i][j] for i in range(m)] for j in range(n)]
        # V[j] = column j of the accumulator V; V starts as I so AV = A.
        V = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]

        eps = sys.float_info.epsilon
        for _ in range(100):
            changed = False
            for p in range(n - 1):
                for q in range(p + 1, n):
                    dot_pp = sum(B[p][i] * B[p][i] for i in range(m))
                    dot_qq = sum(B[q][i] * B[q][i] for i in range(m))
                    dot_pq = sum(B[p][i] * B[q][i] for i in range(m))
                    if abs(dot_pq) <= eps * math.sqrt(dot_pp * dot_qq + eps):
                        continue
                    changed = True
                    tau = (dot_qq - dot_pp) / (2.0 * dot_pq)
                    t = math.copysign(1.0, tau) / (
                        abs(tau) + math.sqrt(1.0 + tau * tau)
                    )
                    c = 1.0 / math.sqrt(1.0 + t * t)
                    s = t * c
                    for i in range(m):
                        bp, bq = B[p][i], B[q][i]
                        B[p][i] = c * bp - s * bq
                        B[q][i] = s * bp + c * bq
                    for i in range(n):
                        vp, vq = V[p][i], V[q][i]
                        V[p][i] = c * vp - s * vq
                        V[q][i] = s * vp + c * vq
            if not changed:
                break

        s_vals = [math.sqrt(sum(B[j][i] * B[j][i] for i in range(m))) for j in range(n)]
        order = sorted(range(n), key=lambda j: s_vals[j], reverse=True)

        s_sorted = [s_vals[order[j]] for j in range(n)]
        B_ord = [B[order[j]] for j in range(n)]
        V_ord = [V[order[j]] for j in range(n)]

        # U[:,j] = B_ord[j] / s_sorted[j], stored row-major.
        U_data = [
            [B_ord[j][i] / s_sorted[j] if s_sorted[j] > eps else 0.0 for j in range(n)]
            for i in range(m)
        ]
        # Vt[j,:] = right singular vector j = column order[j] of V.
        Vt_data = [list(V_ord[j]) for j in range(n)]

        return Tensor(U_data), Tensor(s_sorted), Tensor(Vt_data)

    @classmethod
    def zeros(cls, *shape: int) -> Tensor:
        """Zero Tensor of the given shape (any rank, including rank-0)."""
        if not shape:
            return cls(0.0)

        def _make(dims: tuple[int, ...]) -> list[Any]:
            if len(dims) == 1:
                return [0.0] * dims[0]
            return [_make(dims[1:]) for _ in range(dims[0])]

        return cls(_make(shape))

    @classmethod
    def eye(cls, n: int) -> Tensor:
        """n × n identity matrix."""
        return cls([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])


# ---------------------------------------------------------------------------
# Module-level helpers — operate on raw data (scalar or nested list).
# ---------------------------------------------------------------------------


def _to_list(data: Sequence[Any]) -> list[Any]:
    """Recursively convert nested sequences to nested lists."""
    if not data:
        return []
    if isinstance(data[0], Sequence) and not isinstance(data[0], str | bytes):
        return [_to_list(item) for item in data]
    return list(data)


def _infer_shape(data: list[Any]) -> tuple[int, ...]:
    if not data:
        return (0,)
    if isinstance(data[0], list):
        return (len(data),) + _infer_shape(data[0])
    return (len(data),)


def _map(data: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply fn to every leaf element; data may be a scalar or nested list."""
    if not isinstance(data, list):
        return fn(data)
    if not data or not isinstance(data[0], list):
        return [fn(x) for x in data]
    return [_map(row, fn) for row in data]


def _zip_map(a: Any, b: Any, fn: Callable[[Any, Any], Any]) -> Any:
    """Apply fn element-wise to two same-shape values (scalar or nested list)."""
    if not isinstance(a, list):
        return fn(a, b)
    if not a or not isinstance(a[0], list):
        return [fn(x, y) for x, y in zip(a, b, strict=False)]
    return [_zip_map(ra, rb, fn) for ra, rb in zip(a, b, strict=False)]


def _flatten(data: Any) -> list[float]:
    """Return all leaf elements as a flat list of floats."""
    if not isinstance(data, list):
        return [float(data)]
    if not data:
        return []
    if isinstance(data[0], list):
        result: list[float] = []
        for row in data:
            result.extend(_flatten(row))
        return result
    return [float(x) for x in data]


def _deep_copy(data: Any) -> Any:
    """Deep copy nested lists; return scalars as-is."""
    if not isinstance(data, list):
        return data
    if not data or not isinstance(data[0], list):
        return list(data)
    return [_deep_copy(row) for row in data]


def _matmul_spec(r1: int, r2: int) -> str:
    """Build an einsum spec that contracts the last axis of a rank-r1 tensor
    with the first axis of a rank-r2 tensor."""
    self_idx = "".join(chr(ord("a") + i) for i in range(r1))
    contract = self_idx[-1]
    other_free = "".join(chr(ord("a") + r1 + i) for i in range(r2 - 1))
    return f"{self_idx},{contract}{other_free}->{self_idx[:-1]}{other_free}"


def einsum(spec: str, *tensors: Tensor) -> Tensor:
    """General Einstein summation over one or more Tensors.

    Parameters
    ----------
    spec:
        Subscript string in the form ``'ij,jk->ik'``.  Each comma-separated
        group of letters names the axes of the corresponding tensor; the
        right-hand side lists the axes of the output.  Letters absent from
        the output are contracted (summed) over.
    *tensors:
        One or more Tensor operands whose ranks match the subscript groups.

    Returns
    -------
    Tensor
        Rank equals the number of letters in the output spec (rank-0 when the
        output is empty, e.g. ``'ij,ij->'``).

    Examples
    --------
    Dot product:          ``einsum('i,i->', a, b)``
    Matrix–vector:        ``einsum('ij,j->i', A, x)``
    Matrix multiply:      ``einsum('ij,jk->ik', A, B)``
    Trace:                ``einsum('ii->', A)``
    Outer product:        ``einsum('i,j->ij', a, b)``
    """
    spec = spec.replace(" ", "")
    lhs, out_spec = spec.split("->")
    in_specs = lhs.split(",")

    sizes: dict[str, int] = {}
    for t, s in zip(tensors, in_specs, strict=False):
        for pos, ch in enumerate(s):
            sizes[ch] = t.shape[pos]

    out_chars = list(out_spec)
    contracted = [ch for ch in sizes if ch not in set(out_spec)]
    idx: dict[str, int] = {}

    def _get(t: Tensor, s: str) -> float:
        val: Any = t._data
        for ch in s:
            val = val[idx[ch]]
        return float(val)

    def _sum_contracted(depth: int) -> float:
        if depth == len(contracted):
            return math.prod(
                _get(t, s) for t, s in zip(tensors, in_specs, strict=False)
            )
        ch = contracted[depth]
        total = 0.0
        for i in range(sizes[ch]):
            idx[ch] = i
            total += _sum_contracted(depth + 1)
        return total

    def _build(depth: int) -> Any:
        if depth == len(out_chars):
            return _sum_contracted(0)
        ch = out_chars[depth]
        result = []
        for i in range(sizes[ch]):
            idx[ch] = i
            result.append(_build(depth + 1))
        return result

    return Tensor(_build(0))


__all__ = ["Real", "Tensor", "einsum"]
