"""Tensor: a numeric array of arbitrary rank with a pluggable backend.

The public API is backend-independent.  Operations are dispatched to the
per-instance backend, defaulting to ``PythonBackend`` (nested Python lists).
Switch to ``NumpyBackend`` for BLAS-accelerated computation.

Construction:
    Tensor(1.0)                        — rank-0 scalar
    Tensor([1.0, 2.0, 3.0])            — rank-1, shape (3,)
    Tensor([[1.0, 2.0], [3.0, 4.0]])   — rank-2, shape (2, 2)
    Tensor.zeros()                      — rank-0 zero
    Tensor.zeros(m, n, k)              — rank-3 zero array
    Tensor.eye(n)                       — n × n identity
    Tensor.declare(m, n)               — unallocated tensor, shape (m, n) only

Backend selection:
    Tensor([1.0, 2.0], backend=NumpyBackend())  — per-instance override
    set_default_backend(NumpyBackend())          — process-wide default
    t.to(NumpyBackend())                         — convert existing Tensor

Allocated vs unallocated:
    Tensor.declare(m, n)    — unallocated: shape is known, no memory allocated
    t.is_allocated          — True when values have been allocated
    t.get()                 — extract Python scalar from rank-0 Tensor;
                              raises MaterializationError if unallocated

Indexing (rank ≥ 1 only):
    t[i]          — rank-(n-1) Tensor
    t[i, j, ...]  — rank-(n-k) Tensor when k indices supplied
    t.set(i, v)   — functional write: return new Tensor with position i set to v

Arithmetic (element-wise, any rank):
    a + b, a - b, -a           — element-wise Tensor ± Tensor
    a * b, a / b               — element-wise Tensor × Tensor (Hadamard)
    a * scalar, scalar * a, a / scalar

Tensor contraction:
    einsum(spec, a, b, ...)  — general Einstein summation
    a @ b  — fast-path delegate: dot / vecmat / matvec / matmul

Linear algebra (rank-2 only):
    t.diag()   — main diagonal → rank-1

Any rank:
    t.copy()     — deep copy (same backend)
    t.norm()     — Frobenius norm → rank-0 Tensor
    t.to_list()  — underlying data as a Python scalar or nested list
    t.to(b)      — new Tensor with the same data on backend b
    t.backend    — the backend instance for this Tensor
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from cosmic_foundry.computation.backends import (
    Backend,
    get_default_backend,
)


class MaterializationError(Exception):
    """Raised when attempting to read a value from an unallocated Tensor."""


# ---------------------------------------------------------------------------
# Shape-propagation helpers (used by _DeclaredBackend and module functions)
# ---------------------------------------------------------------------------


def _broadcast(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
    if not s1:
        return s2
    if not s2:
        return s1
    return s1 if len(s1) >= len(s2) else s2


def _slice_output_shape(idx: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(idx, slice):
        n = shape[0] if shape else 0
        return (len(range(*idx.indices(n))),) + shape[1:]
    if isinstance(idx, tuple):
        result: list[int] = []
        remaining = list(shape)
        for i in idx:
            if isinstance(i, int):
                if remaining:
                    remaining.pop(0)
            elif isinstance(i, slice):
                if remaining:
                    n = remaining.pop(0)
                    result.append(len(range(*i.indices(n))))
        result.extend(remaining)
        return tuple(result)
    return shape[1:] if shape else ()


def _matmul_output_shape(
    shape_a: tuple[int, ...], shape_b: tuple[int, ...]
) -> tuple[int, ...]:
    r1, r2 = len(shape_a), len(shape_b)
    if r1 == 1 and r2 == 1:
        return ()
    if r1 == 1 and r2 == 2:
        return (shape_b[1],)
    if r2 == 1:
        return shape_a[:-1]
    return shape_a[:-1] + (shape_b[-1],)


def _einsum_output_shape(spec: str, shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    spec = spec.replace(" ", "")
    lhs, out_spec = spec.split("->")
    in_specs = lhs.split(",")
    sizes: dict[str, int] = {}
    for s, shape in zip(in_specs, shapes, strict=False):
        for pos, ch in enumerate(s):
            sizes[ch] = shape[pos]
    return tuple(sizes[ch] for ch in out_spec)


def _infer_data_shape(data: Any) -> tuple[int, ...]:
    if isinstance(data, list | tuple) and not isinstance(data, bool):
        if not data:
            return (0,)
        return (len(data),) + _infer_data_shape(data[0])
    return ()


def _has_slice(idx: Any) -> bool:
    if isinstance(idx, slice):
        return True
    if isinstance(idx, tuple):
        return any(isinstance(i, slice) for i in idx)
    return False


# ---------------------------------------------------------------------------
# Private backend for declared (unallocated) tensors
# Raw data = shape tuple; every operation propagates shapes.
# ---------------------------------------------------------------------------


class _DeclaredBackend:
    """Internal backend whose raw data is a shape tuple.

    All operations return the output shape (also a tuple) rather than
    computing values.  item() raises MaterializationError so any algorithm
    that calls .get() on an unallocated Tensor is caught immediately.
    """

    def to_native(self, data: Any) -> tuple[int, ...]:
        return _infer_data_shape(data)

    def from_native(self, raw: Any) -> Any:
        return raw

    def infer_shape(self, raw: Any) -> tuple[int, ...]:
        return raw if isinstance(raw, tuple) else ()

    def copy(self, raw: Any) -> tuple[int, ...]:
        return raw  # type: ignore[no-any-return]

    def zeros(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return shape

    def eye(self, n: int) -> tuple[int, ...]:
        return (n, n)

    def get(self, raw: Any) -> Any:
        raise MaterializationError(
            "get() called on an unallocated Tensor: "
            "allocate the Tensor before extracting its value."
        )

    def flatten(self, raw: Any) -> list[float]:
        raise MaterializationError("flatten() called on an unallocated Tensor.")

    def norm(self, raw: Any) -> tuple[int, ...]:
        return ()

    def neg(self, a: Any) -> tuple[int, ...]:
        return a  # type: ignore[no-any-return]

    def add(self, a: Any, b: Any) -> tuple[int, ...]:
        return _broadcast(a, b)

    def sub(self, a: Any, b: Any) -> tuple[int, ...]:
        return _broadcast(a, b)

    def mul_scalar(self, a: Any, s: float) -> tuple[int, ...]:
        return a  # type: ignore[no-any-return]

    def mul_elem(self, a: Any, b: Any) -> tuple[int, ...]:
        return _broadcast(a, b)

    def div_scalar(self, a: Any, s: float) -> tuple[int, ...]:
        return a  # type: ignore[no-any-return]

    def div_elem(self, a: Any, b: Any) -> tuple[int, ...]:
        return _broadcast(a, b)

    def rdiv_scalar(self, s: float, raw: Any) -> tuple[int, ...]:
        return raw  # type: ignore[no-any-return]

    def abs(self, raw: Any) -> tuple[int, ...]:
        return raw  # type: ignore[no-any-return]

    def lt(self, a: Any, b: Any) -> tuple[int, ...]:
        a_s = a if isinstance(a, tuple) else ()
        b_s = b if isinstance(b, tuple) else ()
        return _broadcast(a_s, b_s)

    def le(self, a: Any, b: Any) -> tuple[int, ...]:
        return self.lt(a, b)

    def gt(self, a: Any, b: Any) -> tuple[int, ...]:
        return self.lt(a, b)

    def ge(self, a: Any, b: Any) -> tuple[int, ...]:
        return self.lt(a, b)

    def logical_not(self, raw: Any) -> tuple[int, ...]:
        return raw  # type: ignore[no-any-return]

    def logical_or(self, a: Any, b: Any) -> tuple[int, ...]:
        return self.lt(a, b)

    def where(self, cond: Any, x: Any, y: Any) -> tuple[int, ...]:
        c_s = cond if isinstance(cond, tuple) else ()
        x_s = x if isinstance(x, tuple) else ()
        y_s = y if isinstance(y, tuple) else ()
        return _broadcast(_broadcast(c_s, x_s), y_s)

    def reduce_max(self, raw: Any) -> tuple[int, ...]:
        return ()

    def argmax(self, raw: Any) -> tuple[int, ...]:
        return ()

    def matmul(
        self,
        a: Any,
        b: Any,
        shape_a: tuple[int, ...],
        shape_b: tuple[int, ...],
    ) -> tuple[int, ...]:
        return _matmul_output_shape(shape_a, shape_b)

    def einsum(
        self,
        spec: str,
        raws: list[Any],
        shapes: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        return _einsum_output_shape(spec, shapes)

    def diag(self, raw: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (min(shape),)

    def svd(
        self, raw: Any, shape: tuple[int, ...]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        m, n = shape
        k = min(m, n)
        return (m, k), (k,), (k, k)

    def slice_get(self, raw: Any, idx: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
        return _slice_output_shape(idx, shape)

    def slice_set(
        self, raw: Any, idx: Any, value: Any, shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        return shape

    def take(self, raw: Any, indices: Any) -> tuple[int, ...]:
        return indices if isinstance(indices, tuple) else ()

    def arange(self, n: int) -> tuple[int, ...]:
        return (n,)

    def fori_loop(self, n: int, body_fn: Any, init_state: Any) -> Any:
        return init_state

    def while_loop(self, cond_fn: Any, body_fn: Any, init_state: Any) -> Any:
        return init_state


_DECLARED: _DeclaredBackend = _DeclaredBackend()

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Real protocol check
# ---------------------------------------------------------------------------


@runtime_checkable
class Real(Protocol):
    """Protocol for scalar numeric types usable as Tensor elements."""

    def __neg__(self) -> Real: ...
    def __add__(self, other: Any) -> Real: ...
    def __radd__(self, other: Any) -> Real: ...
    def __sub__(self, other: Any) -> Real: ...
    def __rsub__(self, other: Any) -> Real: ...
    def __mul__(self, other: Any) -> Real: ...
    def __rmul__(self, other: Any) -> Real: ...
    def __truediv__(self, other: Any) -> Real: ...
    def __float__(self) -> float: ...


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


class Tensor(Generic[T]):
    """A numeric array of arbitrary rank backed by a pluggable backend.

    A Tensor is either *allocated* (has a shape and values) or *declared*
    (has a shape only, no memory).  Declared tensors support all shape-
    propagating operations but raise MaterializationError on .get().

    Use Tensor.declare(*shape) to construct a declared Tensor; all standard
    constructors (Tensor(...), Tensor.zeros(...), etc.) produce allocated ones.
    """

    __slots__ = ("_value", "_backend")

    def __init__(self, data: Any, *, backend: Backend | None = None) -> None:
        b: Backend = backend if backend is not None else get_default_backend()
        self._backend: Any = b
        self._value: Any = b.to_native(data)

    @classmethod
    def _wrap(cls, raw: Any, backend: Any) -> Tensor:
        """Construct a Tensor directly from raw backend data."""
        t: Tensor = cls.__new__(cls)
        t._value = raw
        t._backend = backend
        return t

    @classmethod
    def declare(cls, *shape: int) -> Tensor:
        """Declare an unallocated Tensor of the given shape.

        The resulting Tensor propagates shapes through all operations but
        raises MaterializationError on .get().  Pass it to a solver or other
        computation to verify that no value is accidentally extracted before
        the intended materialization boundary.
        """
        t: Tensor = cls.__new__(cls)
        t._value = shape
        t._backend = _DECLARED
        return t

    # ------------------------------------------------------------------
    # Shape and allocation state
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self._value, tuple):
            return self._value
        return self._backend.infer_shape(self._value)  # type: ignore[no-any-return]

    @property
    def is_allocated(self) -> bool:
        return not isinstance(self._value, tuple)

    @property
    def backend(self) -> Backend:
        return self._backend  # type: ignore[no-any-return]

    def to(self, backend: Backend) -> Tensor:
        """Return a new allocated Tensor with the same data on the given backend."""
        if backend is self._backend:
            return Tensor._wrap(self._backend.copy(self._value), backend)
        return Tensor(self._backend.from_native(self._value), backend=backend)

    # ------------------------------------------------------------------
    # Scalar extraction (materialization boundary)
    # ------------------------------------------------------------------

    def get(self) -> T:
        """Extract the Python value from a rank-0 Tensor.

        This is the only sanctioned exit from tensor land into Python land.
        Raises MaterializationError for unallocated Tensors, making every
        call site auditable: grep for .get() to find all materialization
        boundaries in algorithm code.
        """
        if not self.is_allocated:
            raise MaterializationError(
                "get() called on an unallocated Tensor: "
                "allocate the Tensor before extracting its value."
            )
        if self.shape:
            raise TypeError(f"get() requires rank-0 Tensor; got shape {self.shape}")
        return self._backend.get(self._value)  # type: ignore[no-any-return]

    def sync(self) -> None:
        """Block until all pending computation producing this Tensor is complete.

        No-op for synchronous backends (Python, NumPy).  For async backends
        (JAX/XLA, CUDA) this ensures outstanding dispatches have completed
        before the caller proceeds.  Unallocated Tensors are silently ignored.
        """
        if self.is_allocated:
            self._backend.sync(self._value)

    def __float__(self) -> float:
        return float(self.get())  # type: ignore[arg-type]

    def __bool__(self) -> bool:
        return bool(self.get())

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("rank-0 Tensor has no length")
        return self.shape[0]

    def __getitem__(self, idx: Any) -> Tensor:
        if not self.shape:
            raise TypeError("rank-0 Tensor is not subscriptable")
        if isinstance(idx, Tensor):
            if not self.is_allocated or not idx.is_allocated:
                return Tensor._wrap(self.shape[1:], self._backend)
            return Tensor._wrap(self._value[idx._value], self._backend)
        if not self.is_allocated:
            if _has_slice(idx):
                out = self._backend.slice_get(self._value, idx, self.shape)
            elif isinstance(idx, tuple):
                out = self._value
                for i in idx:
                    if isinstance(i, slice):
                        n = out[0] if out else 0
                        length = len(range(*i.indices(n)))
                        out = (length,) + out[1:]
                    else:
                        out = out[1:] if out else ()
            else:
                out = self.shape[1:]
            return Tensor._wrap(out, self._backend)
        if _has_slice(idx):
            raw = self._backend.slice_get(self._value, idx, self.shape)
            return Tensor._wrap(raw, self._backend)
        if isinstance(idx, tuple):
            result: Any = self
            for i in idx:
                if not isinstance(result, Tensor):
                    raise IndexError("too many indices for Tensor")
                result = result[i]
            return result  # type: ignore[no-any-return]
        return Tensor._wrap(self._value[idx], self._backend)

    def set(self, idx: Any, value: Any) -> Tensor:
        """Return a new Tensor with position idx set to value (functional write).

        For unallocated Tensors, returns a new unallocated Tensor of the same
        shape (shape is preserved by any write).
        """
        if not self.is_allocated:
            return Tensor._wrap(self.shape, self._backend)
        v = value._value if isinstance(value, Tensor) else value
        raw_idx = idx._value if isinstance(idx, Tensor) else idx
        new_raw = self._backend.slice_set(self._value, raw_idx, v, self.shape)
        return Tensor._wrap(new_raw, self._backend)

    def __iter__(self) -> Iterator[Any]:
        if not self.shape:
            raise TypeError("rank-0 Tensor is not iterable")
        for i in range(self.shape[0]):
            yield self[i]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> Tensor:
        """Return a deep copy of this Tensor (same backend)."""
        return Tensor._wrap(self._backend.copy(self._value), self._backend)

    def to_list(self) -> Any:
        """Scalar for rank-0; deep-copied nested list for rank ≥ 1."""
        if not self.is_allocated:
            raise MaterializationError("to_list() called on an unallocated Tensor.")
        return self._backend.from_native(self._value)

    def __repr__(self) -> str:
        if not self.is_allocated:
            return f"Tensor.declare{self.shape}"
        return f"Tensor({self._backend.from_native(self._value)!r})"

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def _check_backend(self, other: Tensor) -> None:
        if self._backend is not other._backend:
            raise ValueError(
                f"cannot mix backends: {type(self._backend).__name__} "
                f"and {type(other._backend).__name__}"
            )

    def __neg__(self) -> Tensor:
        return Tensor._wrap(self._backend.neg(self._value), self._backend)

    def __add__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            self._check_backend(other)
            other_raw = other._value
        else:
            other_raw = self._backend.to_native(other)
        return Tensor._wrap(self._backend.add(self._value, other_raw), self._backend)

    def __radd__(self, other: Any) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            self._check_backend(other)
            other_raw = other._value
        else:
            other_raw = self._backend.to_native(other)
        return Tensor._wrap(self._backend.sub(self._value, other_raw), self._backend)

    def __rsub__(self, other: Any) -> Tensor:
        other_raw = self._backend.to_native(other)
        return Tensor._wrap(self._backend.sub(other_raw, self._value), self._backend)

    def __mul__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            self._check_backend(other)
            return Tensor._wrap(
                self._backend.mul_elem(self._value, other._value), self._backend
            )
        return Tensor._wrap(
            self._backend.mul_scalar(self._value, float(other)), self._backend
        )

    def __rmul__(self, other: Any) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            self._check_backend(other)
            return Tensor._wrap(
                self._backend.div_elem(self._value, other._value), self._backend
            )
        return Tensor._wrap(
            self._backend.div_scalar(self._value, float(other)), self._backend
        )

    def __rtruediv__(self, other: float) -> Tensor:
        return Tensor._wrap(
            self._backend.rdiv_scalar(float(other), self._value), self._backend
        )

    def __lt__(self, other: Any) -> Tensor:
        other_raw = (
            other._value
            if isinstance(other, Tensor)
            else self._backend.to_native(other)
        )
        return Tensor._wrap(self._backend.lt(self._value, other_raw), self._backend)

    def __le__(self, other: Any) -> Tensor:
        other_raw = (
            other._value
            if isinstance(other, Tensor)
            else self._backend.to_native(other)
        )
        return Tensor._wrap(self._backend.le(self._value, other_raw), self._backend)

    def __gt__(self, other: Any) -> Tensor:
        other_raw = (
            other._value
            if isinstance(other, Tensor)
            else self._backend.to_native(other)
        )
        return Tensor._wrap(self._backend.gt(self._value, other_raw), self._backend)

    def __ge__(self, other: Any) -> Tensor:
        other_raw = (
            other._value
            if isinstance(other, Tensor)
            else self._backend.to_native(other)
        )
        return Tensor._wrap(self._backend.ge(self._value, other_raw), self._backend)

    def __invert__(self) -> Tensor:
        return Tensor._wrap(self._backend.logical_not(self._value), self._backend)

    def __or__(self, other: Any) -> Tensor:
        other_raw = (
            other._value
            if isinstance(other, Tensor)
            else self._backend.to_native(other)
        )
        return Tensor._wrap(
            self._backend.logical_or(self._value, other_raw), self._backend
        )

    def __matmul__(self, other: Tensor) -> Tensor:
        self._check_backend(other)
        raw = self._backend.matmul(self._value, other._value, self.shape, other.shape)
        return Tensor._wrap(raw, self._backend)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def abs(self) -> Tensor:
        return Tensor._wrap(self._backend.abs(self._value), self._backend)

    def max(self) -> Tensor:
        """Maximum of all elements; returns a 0-d scalar Tensor."""
        return Tensor._wrap(self._backend.reduce_max(self._value), self._backend)

    def argmax(self) -> Tensor[int]:
        """Index of the maximum element in a 1-d Tensor; returns rank-0 Tensor[int]."""
        return Tensor._wrap(self._backend.argmax(self._value), self._backend)

    def element(self, *indices: int) -> Tensor:
        """Return the scalar at the given static integer indices as a 0-d Tensor."""
        if not self.is_allocated:
            return Tensor._wrap((), self._backend)
        raw = self._value
        for i in indices:
            raw = raw[i]
        return Tensor._wrap(raw, self._backend)

    def take(self, indices: Tensor) -> Tensor:
        """Gather elements at integer indices; return a new Tensor."""
        self._check_backend(indices)
        return Tensor._wrap(
            self._backend.take(self._value, indices._value), self._backend
        )

    def diag(self) -> Tensor:
        """Main diagonal of a rank-2 Tensor → rank-1."""
        if len(self.shape) != 2:
            raise ValueError(f"diag requires rank-2 Tensor, got shape {self.shape}")
        return Tensor._wrap(self._backend.diag(self._value, self.shape), self._backend)

    def norm(self) -> Tensor:
        """Frobenius norm (absolute value for rank-0) → rank-0 Tensor."""
        if not self.shape:
            return Tensor._wrap(self._backend.abs(self._value), self._backend)
        return Tensor._wrap(self._backend.norm(self._value), self._backend)

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def zeros(cls, *shape: int, backend: Backend | None = None) -> Tensor:
        """Zero Tensor of the given shape (any rank, including rank-0)."""
        b: Any = backend if backend is not None else get_default_backend()
        return cls._wrap(b.zeros(shape), b)

    @classmethod
    def eye(cls, n: int, *, backend: Backend | None = None) -> Tensor:
        """n × n identity matrix."""
        b: Any = backend if backend is not None else get_default_backend()
        return cls._wrap(b.eye(n), b)


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


def einsum(spec: str, *tensors: Tensor) -> Tensor:
    """General Einstein summation over one or more Tensors."""
    if not tensors:
        raise ValueError("einsum requires at least one tensor")
    backend = tensors[0]._backend
    for t in tensors[1:]:
        if t._backend is not backend:
            raise ValueError(
                f"all tensors in einsum must use the same backend; "
                f"got {type(backend).__name__} and {type(t._backend).__name__}"
            )
    raws = [t._value for t in tensors]
    shapes = [t.shape for t in tensors]
    return Tensor._wrap(backend.einsum(spec, raws, shapes), backend)


def where(cond: Tensor, x: Any, y: Any) -> Tensor:
    """Element-wise ternary: return x where cond is True, y elsewhere."""
    backend = cond._backend
    x_raw = x._value if isinstance(x, Tensor) else backend.to_native(x)
    y_raw = y._value if isinstance(y, Tensor) else backend.to_native(y)
    return Tensor._wrap(backend.where(cond._value, x_raw, y_raw), backend)


def arange(n: int, *, backend: Backend | None = None) -> Tensor[int]:
    """Return a 1-d integer Tensor [0, 1, ..., n-1]."""
    b: Any = backend if backend is not None else get_default_backend()
    return Tensor._wrap(b.arange(n), b)


__all__ = ["MaterializationError", "Real", "Tensor", "arange", "einsum", "where"]
