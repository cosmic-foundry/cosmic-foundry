"""Tensor: a pure-Python rank-1 / rank-2 numeric array.

All arithmetic is implemented with Python lists and the math module.
No NumPy or JAX is used.  This is the reference implementation — kept
permanently for readability and cross-checking against higher-performance
backends.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator


class Tensor:
    """A rank-1 or rank-2 numeric array backed by nested Python lists.

    Supports the arithmetic and linear algebra operations needed by the
    computation layer: addition, scaling, matrix multiply, norm, SVD.
    All operations return new Tensor objects; the original is never mutated.

    Construction:
        Tensor([1.0, 2.0, 3.0])           — rank-1, shape (3,)
        Tensor([[1.0, 2.0], [3.0, 4.0]])  — rank-2, shape (2, 2)
        Tensor.zeros(m, n)                — rank-2 zero matrix
        Tensor.eye(n)                     — n × n identity

    Indexing:
        t[i]       — float for rank-1, rank-1 Tensor (row i) for rank-2
        t[i][j]    — float via chain for rank-2

    Arithmetic:
        a + b, a - b        — element-wise, same shape
        -a                  — element-wise negation
        a * scalar          — element-wise scaling
        a / scalar          — element-wise scaling by reciprocal
        a @ b               — dot (rank-1 @ rank-1 → float),
                              matvec (rank-2 @ rank-1 → rank-1),
                              matmul (rank-2 @ rank-2 → rank-2)

    Linear algebra:
        t.diag()            — rank-2 → rank-1: main diagonal
        t.norm()            — L² norm (rank-1) or Frobenius norm (rank-2) → float
        t.svd()             — rank-2 → (U, s, Vt); singular values descending
    """

    def __init__(self, data: list) -> None:
        self._data = data
        self._shape: tuple[int, ...] = _infer_shape(data)

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

    def to_list(self) -> list:
        """Return a copy of the underlying nested Python list."""
        if len(self._shape) == 1:
            return list(self._data)
        return [list(row) for row in self._data]

    def __repr__(self) -> str:
        return f"Tensor({self._data!r})"

    def __neg__(self) -> Tensor:
        if len(self._shape) == 1:
            return Tensor([-x for x in self._data])
        return Tensor([[-x for x in row] for row in self._data])

    def __add__(self, other: Tensor) -> Tensor:
        if len(self._shape) == 1:
            return Tensor(
                [self._data[i] + other._data[i] for i in range(self._shape[0])]
            )
        return Tensor(
            [
                [self._data[i][j] + other._data[i][j] for j in range(self._shape[1])]
                for i in range(self._shape[0])
            ]
        )

    def __sub__(self, other: Tensor) -> Tensor:
        if len(self._shape) == 1:
            return Tensor(
                [self._data[i] - other._data[i] for i in range(self._shape[0])]
            )
        return Tensor(
            [
                [self._data[i][j] - other._data[i][j] for j in range(self._shape[1])]
                for i in range(self._shape[0])
            ]
        )

    def __mul__(self, scalar: float) -> Tensor:
        s = float(scalar)
        if len(self._shape) == 1:
            return Tensor([x * s for x in self._data])
        return Tensor([[x * s for x in row] for row in self._data])

    def __rmul__(self, scalar: float) -> Tensor:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Tensor:
        return self.__mul__(1.0 / float(scalar))

    def __matmul__(self, other: Tensor) -> float | Tensor:
        if len(self._shape) == 1 and len(other._shape) == 1:
            return float(
                sum(self._data[i] * other._data[i] for i in range(self._shape[0]))
            )
        if len(self._shape) == 2 and len(other._shape) == 1:
            m, n = self._shape
            return Tensor(
                [
                    sum(self._data[i][j] * other._data[j] for j in range(n))
                    for i in range(m)
                ]
            )
        if len(self._shape) == 2 and len(other._shape) == 2:
            m, k = self._shape
            _, n = other._shape
            return Tensor(
                [
                    [
                        sum(self._data[i][p] * other._data[p][j] for p in range(k))
                        for j in range(n)
                    ]
                    for i in range(m)
                ]
            )
        raise ValueError(f"unsupported matmul: {self._shape} @ {other._shape}")

    def diag(self) -> Tensor:
        """Main diagonal of a rank-2 Tensor → rank-1."""
        if len(self._shape) != 2:
            raise ValueError(f"diag requires rank-2 Tensor, got shape {self._shape}")
        n = min(self._shape)
        return Tensor([self._data[i][i] for i in range(n)])

    def norm(self) -> float:
        """L² norm (rank-1) or Frobenius norm (rank-2)."""
        if len(self._shape) == 1:
            return math.sqrt(sum(x * x for x in self._data))
        return math.sqrt(sum(x * x for row in self._data for x in row))

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
        """Zero Tensor of the given shape (rank 1 or 2)."""
        if len(shape) == 1:
            return cls([0.0] * shape[0])
        if len(shape) == 2:
            return cls([[0.0] * shape[1] for _ in range(shape[0])])
        raise ValueError(f"zeros supports rank 1 or 2, got shape {shape}")

    @classmethod
    def eye(cls, n: int) -> Tensor:
        """n × n identity matrix."""
        return cls([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])


def _infer_shape(data: list) -> tuple[int, ...]:
    if not isinstance(data, list) or not data:
        return (0,)
    if isinstance(data[0], list):
        return (len(data), len(data[0]))
    return (len(data),)


__all__ = ["Tensor"]
