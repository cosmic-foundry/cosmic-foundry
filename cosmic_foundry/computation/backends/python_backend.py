"""Pure-Python backend for Tensor: all operations on nested Python lists."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any


class PythonBackend:
    """Implements all Tensor operations using nested Python lists.

    Leaf elements must satisfy the Real protocol.  This backend is the
    reference implementation — kept permanently for readability and for
    environments where NumPy is unavailable.
    """

    @property
    def min_ops(self) -> int:
        """CPU backend: 1 000 ops is enough for compute to dominate overhead."""
        return 1_000

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def to_native(self, data: Any) -> Any:
        """Convert any scalar or nested sequence to a nested Python list."""
        if isinstance(data, list | tuple) or (
            hasattr(data, "__len__") and not isinstance(data, str | bytes)
        ):
            return _to_list(data)
        return data  # scalar: returned as-is

    def from_native(self, raw: Any) -> Any:
        """Deep-copy nested list (or return scalar) for cross-backend transfer."""
        return _deep_copy(raw)

    def infer_shape(self, raw: Any) -> tuple[int, ...]:
        if not isinstance(raw, list):
            return ()
        return _infer_shape(raw)

    def copy(self, raw: Any) -> Any:
        return _deep_copy(raw)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def zeros(self, shape: tuple[int, ...]) -> Any:
        if not shape:
            return 0.0

        def _make(dims: tuple[int, ...]) -> list[Any]:
            if len(dims) == 1:
                return [0.0] * dims[0]
            return [_make(dims[1:]) for _ in range(dims[0])]

        return _make(shape)

    def eye(self, n: int) -> list[list[float]]:
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def flatten(self, raw: Any) -> list[float]:
        return _flatten(raw)

    def norm(self, raw: Any) -> float:
        return math.sqrt(sum(x * x for x in _flatten(raw)))

    # ------------------------------------------------------------------
    # Element-wise arithmetic
    # ------------------------------------------------------------------

    def neg(self, a: Any) -> Any:
        return _map(a, lambda x: -x)

    def add(self, a: Any, b: Any) -> Any:
        return _zip_map(a, b, lambda x, y: x + y)

    def sub(self, a: Any, b: Any) -> Any:
        return _zip_map(a, b, lambda x, y: x - y)

    def mul_scalar(self, a: Any, s: float) -> Any:
        return _map(a, lambda x: x * s)

    def mul_elem(self, a: Any, b: Any) -> Any:
        return _zip_map(a, b, lambda x, y: x * y)

    def div_scalar(self, a: Any, s: float) -> Any:
        return _map(a, lambda x: x / s)

    def div_elem(self, a: Any, b: Any) -> Any:
        return _zip_map(a, b, lambda x, y: x / y)

    # ------------------------------------------------------------------
    # Reductions and element-wise ops for JIT-clean algorithms
    # ------------------------------------------------------------------

    def abs(self, raw: Any) -> Any:
        return _map(raw, lambda x: abs(x))

    def reduce_max(self, raw: Any) -> Any:
        return max(_flatten(raw))

    def argmax(self, raw: Any) -> int:
        return max(range(len(raw)), key=lambda i: raw[i])

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return _where(cond, x, y)

    def lt(self, a: Any, b: Any) -> Any:
        return _zip_map(a, b, lambda x, y: x < y)

    def le(self, a: Any, b: Any) -> Any:
        return _zip_map(a, b, lambda x, y: x <= y)

    def gt(self, a: Any, b: Any) -> Any:
        return _zip_map(a, b, lambda x, y: x > y)

    def ge(self, a: Any, b: Any) -> Any:
        return _zip_map(a, b, lambda x, y: x >= y)

    def logical_not(self, raw: Any) -> Any:
        if isinstance(raw, list):
            return _map(raw, lambda x: not x)
        return not raw

    def logical_or(self, a: Any, b: Any) -> Any:
        return _zip_map(a, b, lambda x, y: x or y)

    def arange(self, n: int) -> Any:
        return list(range(n))

    def rdiv_scalar(self, s: float, raw: Any) -> Any:
        return _map(raw, lambda x: s / x)

    def take(self, raw: Any, indices: Any) -> Any:
        return [raw[int(i)] for i in _flatten(indices)]

    def get(self, raw: Any) -> Any:
        return raw

    def sync(self, raw: Any) -> None:
        pass

    def fori_loop(
        self,
        n: int,
        body_fn: Callable[[Any, Any], Any],
        init_state: Any,
    ) -> Any:
        state = init_state
        for k in range(n):
            state = body_fn(k, state)
        return state

    def while_loop(
        self,
        cond_fn: Callable[[Any], Any],
        body_fn: Callable[[Any], Any],
        init_state: Any,
    ) -> Any:
        from cosmic_foundry.computation.tensor import Tensor

        state = init_state
        cond = cond_fn(state)
        while bool(cond.get() if isinstance(cond, Tensor) else cond):
            state = body_fn(state)
            cond = cond_fn(state)
        return state

    # ------------------------------------------------------------------
    # Contraction
    # ------------------------------------------------------------------

    def matmul(
        self,
        a: Any,
        b: Any,
        shape_a: tuple[int, ...],
        shape_b: tuple[int, ...],
    ) -> Any:
        r1, r2 = len(shape_a), len(shape_b)
        if r1 == 1 and r2 == 1:
            return sum(a[i] * b[i] for i in range(shape_a[0]))
        if r1 == 1 and r2 == 2:
            k, n = shape_b
            return [sum(a[p] * b[p][j] for p in range(k)) for j in range(n)]
        if r2 == 1:
            return _matvec(a, b)
        if r2 == 2:
            return _matmul_lists(a, b)
        return self.einsum(_matmul_spec(r1, r2), [a, b], [shape_a, shape_b])

    def einsum(
        self,
        spec: str,
        raws: list[Any],
        shapes: list[tuple[int, ...]],
    ) -> Any:
        spec = spec.replace(" ", "")
        lhs, out_spec = spec.split("->")
        in_specs = lhs.split(",")

        sizes: dict[str, int] = {}
        for s, sh in zip(in_specs, shapes, strict=False):
            for pos, ch in enumerate(s):
                sizes[ch] = sh[pos]

        out_chars = list(out_spec)
        contracted = [ch for ch in sizes if ch not in set(out_spec)]
        idx: dict[str, int] = {}

        def _get(raw: Any, s: str) -> float:
            val: Any = raw
            for ch in s:
                val = val[idx[ch]]
            return float(val)

        def _sum_contracted(depth: int) -> float:
            if depth == len(contracted):
                return math.prod(
                    _get(raw, s) for raw, s in zip(raws, in_specs, strict=False)
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

        result = _build(0)
        return result if out_chars else result

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def diag(self, raw: list[Any], shape: tuple[int, ...]) -> list[Any]:
        n = min(shape)
        return [raw[i][i] for i in range(n)]

    # ------------------------------------------------------------------
    # Slice indexing
    # ------------------------------------------------------------------

    def slice_get(self, raw: Any, idx: Any, shape: tuple[int, ...]) -> Any:
        """Read raw[idx]; idx contains at least one slice object."""
        if isinstance(idx, slice):
            return raw[idx]
        if isinstance(idx, tuple) and len(idx) == 2:
            r0, r1 = idx
            if isinstance(r0, int) and isinstance(r1, slice):
                return raw[r0][r1]
            if isinstance(r0, slice) and isinstance(r1, int):
                return [row[r1] for row in raw[r0]]
            if isinstance(r0, slice) and isinstance(r1, slice):
                return [row[r1] for row in raw[r0]]
        raise NotImplementedError(f"PythonBackend.slice_get: unsupported idx {idx!r}")

    def slice_set(self, raw: Any, idx: Any, value: Any, shape: tuple[int, ...]) -> Any:
        """Assign raw[idx] = value in-place; return raw."""
        if isinstance(idx, int | slice):
            raw[idx] = value
            return raw
        if isinstance(idx, tuple):
            if len(idx) == 1:
                raw[idx[0]] = value
                return raw
            if all(isinstance(i, int) for i in idx):
                data = raw
                for i in idx[:-1]:
                    data = data[i]
                data[idx[-1]] = value
                return raw
            if len(idx) == 2:
                r0, r1 = idx
                if isinstance(r0, int) and isinstance(r1, slice):
                    raw[r0][r1] = value
                    return raw
                if isinstance(r0, slice) and isinstance(r1, int):
                    for row, v in zip(raw[r0], value, strict=False):
                        row[r1] = v
                    return raw
                if isinstance(r0, slice) and isinstance(r1, slice):
                    for row, val_row in zip(raw[r0], value, strict=False):
                        row[r1] = val_row
                    return raw
        raise NotImplementedError(f"PythonBackend.slice_set: unsupported idx {idx!r}")

    def svd(self, raw: list[Any], shape: tuple[int, ...]) -> tuple[Any, Any, Any]:
        m, n = shape
        B = [[raw[i][j] for i in range(m)] for j in range(n)]
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

        U_data = [
            [B_ord[j][i] / s_sorted[j] if s_sorted[j] > eps else 0.0 for j in range(n)]
            for i in range(m)
        ]
        Vt_data = [list(V_ord[j]) for j in range(n)]
        return U_data, s_sorted, Vt_data


# ---------------------------------------------------------------------------
# Module-level helpers (pure Python, operate on raw nested lists / scalars)
# ---------------------------------------------------------------------------


def _to_list(data: Any) -> list[Any]:
    if not data:
        return []
    if isinstance(data[0], list | tuple) or (
        hasattr(data[0], "__len__") and not isinstance(data[0], str | bytes)
    ):
        return [_to_list(item) for item in data]
    return list(data)


def _infer_shape(data: list[Any]) -> tuple[int, ...]:
    if not data:
        return (0,)
    if isinstance(data[0], list):
        return (len(data),) + _infer_shape(data[0])
    return (len(data),)


def _map(data: Any, fn: Callable[[Any], Any]) -> Any:
    if not isinstance(data, list):
        return fn(data)
    if not data or not isinstance(data[0], list):
        return [fn(x) for x in data]
    return [_map(row, fn) for row in data]


def _zip_map(a: Any, b: Any, fn: Callable[[Any, Any], Any]) -> Any:
    if not isinstance(a, list) and not isinstance(b, list):
        return fn(a, b)
    if not isinstance(a, list):
        return _map(b, lambda y: fn(a, y))
    if not isinstance(b, list):
        return _map(a, lambda x: fn(x, b))
    if not a or not isinstance(a[0], list):
        return [fn(x, y) for x, y in zip(a, b, strict=False)]
    return [_zip_map(ra, rb, fn) for ra, rb in zip(a, b, strict=False)]


def _flatten(data: Any) -> list[float]:
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


def _where(cond: Any, x: Any, y: Any) -> Any:
    """Element-wise ternary; scalar cond broadcasts over array x/y."""
    if not isinstance(cond, list):
        return x if cond else y
    x_is_list = isinstance(x, list)
    y_is_list = isinstance(y, list)
    result = []
    for i, c in enumerate(cond):
        xi = x[i] if x_is_list else x
        yi = y[i] if y_is_list else y
        result.append(_where(c, xi, yi))
    return result


def _deep_copy(data: Any) -> Any:
    if not isinstance(data, list):
        return data
    if not data or not isinstance(data[0], list):
        return list(data)
    return [_deep_copy(row) for row in data]


def _matvec(a: list[Any], x: list[Any]) -> list[Any]:
    if not isinstance(a[0], list):
        return [sum(a[i] * x[i] for i in range(len(x)))]
    if not isinstance(a[0][0], list):
        return [sum(row[j] * x[j] for j in range(len(x))) for row in a]
    return [_matvec(sub, x) for sub in a]


def _matmul_lists(a: list[Any], b: list[Any]) -> list[Any]:
    if not isinstance(a[0][0], list):
        k, n = len(b), len(b[0])
        return [
            [sum(a[i][p] * b[p][j] for p in range(k)) for j in range(n)]
            for i in range(len(a))
        ]
    return [_matmul_lists(sub, b) for sub in a]


def _matmul_spec(r1: int, r2: int) -> str:
    self_idx = "".join(chr(ord("a") + i) for i in range(r1))
    contract = self_idx[-1]
    other_free = "".join(chr(ord("a") + r1 + i) for i in range(r2 - 1))
    return f"{self_idx},{contract}{other_free}->{self_idx[:-1]}{other_free}"
