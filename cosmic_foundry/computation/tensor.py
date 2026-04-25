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
        Tensor([1.0, 2.0, 3.0])           — rank-1, shape (3,)
        Tensor([[1.0, 2.0], [3.0, 4.0]])  — rank-2, shape (2, 2)
        Tensor([[[...], ...], ...])        — rank-3+, arbitrary shape
        Tensor.zeros(m, n, k)             — rank-3 zero array
        Tensor.eye(n)                     — n × n identity

    Indexing:
        t[i]   — float for rank-1, rank-(n-1) Tensor for rank-n

    Arithmetic (element-wise, any rank):
        a + b, a - b, -a
        a * scalar, scalar * a, a / scalar
        a @ b  — dot      (rank-1 @ rank-1 → float),
                  vecmat   (rank-1 @ rank-2 → rank-1),
                  matvec   (rank-n @ rank-1 → rank-(n-1)),
                  matmul   (rank-n @ rank-2 → rank-n)

    Linear algebra (rank-2 only):
        t.diag()   — main diagonal → rank-1
        t.svd()    — (U, s, Vt); singular values descending

    Any rank:
        t.norm()   — Frobenius norm → float
        t.to_list() — deep copy of the underlying nested list
    """

    def __init__(self, data: Sequence[Any]) -> None:
        # _to_list recursively converts nested sequences to nested lists so
        # that all internal helpers can assume list as the container type.
        self._data: list[Any] = _to_list(data)
        self._shape: tuple[int, ...] = _infer_shape(self._data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, idx: int) -> float | Tensor:
        if len(self._shape) == 1:
            return float(self._data[idx])
        return Tensor(self._data[idx])

    def __iter__(self) -> Iterator[float | Tensor]:
        for i in range(self._shape[0]):
            yield self[i]

    def to_list(self) -> list[Any]:
        """Return a deep copy of the underlying nested Python list."""
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

    def __matmul__(self, other: Tensor) -> float | Tensor:
        r1, r2 = len(self._shape), len(other._shape)
        if r1 == 1 and r2 == 1:
            # dot product
            return float(
                sum(self._data[i] * other._data[i] for i in range(self._shape[0]))
            )
        if r1 == 1 and r2 == 2:
            # vecmat: (k,) @ (k, n) → (n,)
            k, n = other._shape
            return Tensor(
                [
                    sum(self._data[p] * other._data[p][j] for p in range(k))
                    for j in range(n)
                ]
            )
        if r2 == 1:
            # batched matvec: (..., m, k) @ (k,) → (..., m)
            return Tensor(_matvec(self._data, other._data))
        if r2 == 2:
            # batched matmul: (..., m, k) @ (k, n) → (..., m, n)
            return Tensor(_matmul(self._data, other._data))
        raise ValueError(f"unsupported matmul: {self._shape} @ {other._shape}")

    def diag(self) -> Tensor:
        """Main diagonal of a rank-2 Tensor → rank-1."""
        if len(self._shape) != 2:
            raise ValueError(f"diag requires rank-2 Tensor, got shape {self._shape}")
        n = min(self._shape)
        return Tensor([self._data[i][i] for i in range(n)])

    def norm(self) -> float:
        """Frobenius norm: sqrt of sum of squares of all elements."""
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
        """Zero Tensor of the given shape (any rank ≥ 1)."""
        if not shape:
            raise ValueError("zeros requires at least one dimension")

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
# Module-level helpers — operate on raw nested lists, not Tensor objects.
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


def _map(data: list[Any], fn: Callable[[Any], Any]) -> list[Any]:
    """Apply fn to every leaf element of a nested list."""
    if not data or not isinstance(data[0], list):
        return [fn(x) for x in data]
    return [_map(row, fn) for row in data]


def _zip_map(a: list[Any], b: list[Any], fn: Callable[[Any, Any], Any]) -> list[Any]:
    """Apply fn element-wise to two nested lists of the same shape."""
    if not a or not isinstance(a[0], list):
        return [fn(x, y) for x, y in zip(a, b, strict=False)]
    return [_zip_map(ra, rb, fn) for ra, rb in zip(a, b, strict=False)]


def _flatten(data: list[Any]) -> list[float]:
    """Return all leaf elements as a flat list of floats."""
    if not data:
        return []
    if isinstance(data[0], list):
        result: list[float] = []
        for row in data:
            result.extend(_flatten(row))
        return result
    return [float(x) for x in data]


def _deep_copy(data: list[Any]) -> list[Any]:
    """Deep copy a nested list."""
    if not data or not isinstance(data[0], list):
        return list(data)
    return [_deep_copy(row) for row in data]


def _matvec(a: list[Any], x: list[Any]) -> list[Any]:
    """Batched matvec: a has shape (..., m, k), x has shape (k,) → (..., m)."""
    if not isinstance(a[0][0], list):
        # a is rank-2
        return [sum(row[j] * x[j] for j in range(len(x))) for row in a]
    return [_matvec(sub, x) for sub in a]


def _matmul(a: list[Any], b: list[Any]) -> list[Any]:
    """Batched matmul: a has shape (..., m, k), b has shape (k, n) → (..., m, n)."""
    if not isinstance(a[0][0], list):
        # a is rank-2
        k, n = len(b), len(b[0])
        return [
            [sum(a[i][p] * b[p][j] for p in range(k)) for j in range(n)]
            for i in range(len(a))
        ]
    return [_matmul(sub, b) for sub in a]


__all__ = ["Real", "Tensor"]
