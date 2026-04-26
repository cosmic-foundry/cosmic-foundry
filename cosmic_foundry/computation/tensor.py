"""Tensor: a numeric array of arbitrary rank with a pluggable backend.

The public API is backend-independent.  Operations are dispatched to the
per-instance backend, defaulting to ``PythonBackend`` (nested Python lists).
Switch to ``NumpyBackend`` for BLAS-accelerated computation.

Construction:
    Tensor(1.0)                        — rank-0 scalar
    Tensor([1.0, 2.0, 3.0])            — rank-1, shape (3,)
    Tensor([[1.0, 2.0], [3.0, 4.0]])   — rank-2, shape (2, 2)
    Tensor([[[...], ...], ...])         — rank-3+, arbitrary shape
    Tensor.zeros()                      — rank-0 zero
    Tensor.zeros(m, n, k)              — rank-3 zero array
    Tensor.eye(n)                       — n × n identity

Backend selection:
    Tensor([1.0, 2.0], backend=NumpyBackend())  — per-instance override
    set_default_backend(NumpyBackend())          — process-wide default
    t.to(NumpyBackend())                         — convert existing Tensor

Indexing (rank ≥ 1 only):
    t[i]          — Real for rank-1, rank-(n-1) Tensor for rank-n
    t[i, j, ...]  — Real when all indices supplied, Tensor otherwise
    t[i] = v      — assign element (Real) or sub-tensor to position i
    t[i, j] = v   — assign element or sub-tensor at multi-index

Arithmetic (element-wise, any rank):
    a + b, a - b, -a           — element-wise Tensor ± Tensor
    a * b, a / b               — element-wise Tensor × Tensor (Hadamard)
    a * scalar, scalar * a, a / scalar
    float(t)  — extract value from rank-0 Tensor

Tensor contraction:
    einsum(spec, a, b, ...)  — general Einstein summation
    a @ b  — fast-path delegate: dot / vecmat / matvec / matmul

Linear algebra (rank-2 only):
    t.diag()   — main diagonal → rank-1
    t.svd()    — (U, s, Vt); singular values descending

Any rank:
    t.copy()     — deep copy (same backend)
    t.norm()     — Frobenius norm (|value| for rank-0) → float
    t.to_list()  — underlying data as a Python scalar or nested list
    t.to(b)      — new Tensor with the same data on backend b
    t.backend    — the backend instance for this Tensor
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from cosmic_foundry.computation.backends import (
    Backend,
    get_default_backend,
)


def _has_slice(idx: Any) -> bool:
    """Return True if idx is or contains a slice object."""
    if isinstance(idx, slice):
        return True
    if isinstance(idx, tuple):
        return any(isinstance(i, slice) for i in idx)
    return False


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
    """A numeric array of arbitrary rank backed by a pluggable backend."""

    __slots__ = ("_data", "_shape", "_backend")

    def __init__(self, data: Any, *, backend: Backend | None = None) -> None:
        self._backend: Backend = (
            backend if backend is not None else get_default_backend()
        )
        self._data: Any = self._backend.to_native(data)
        self._shape: tuple[int, ...] = self._backend.infer_shape(self._data)

    @classmethod
    def _wrap(cls, raw: Any, backend: Backend) -> Tensor:
        """Construct a Tensor directly from raw backend data, bypassing to_native."""
        t: Tensor = cls.__new__(cls)
        t._data = raw
        t._backend = backend
        t._shape = backend.infer_shape(raw)
        return t

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def backend(self) -> Backend:
        return self._backend

    def to(self, backend: Backend) -> Tensor:
        """Return a new Tensor with the same data on the given backend."""
        if backend is self._backend:
            return Tensor._wrap(self._backend.copy(self._data), backend)
        return Tensor(self._backend.from_native(self._data), backend=backend)

    # ------------------------------------------------------------------
    # Scalar extraction
    # ------------------------------------------------------------------

    def item(self) -> float | bool:
        """Explicitly materialise a rank-0 Tensor into a Python scalar.

        This is the only sanctioned exit from tensor land into Python land.
        TracingBackend raises JitIncompatibleError here, making every call
        site auditable: grep for ``.item()`` to find all materialization
        boundaries in algorithm code.
        """
        if self._shape:
            raise TypeError(f"item() requires rank-0 Tensor; got shape {self._shape}")
        return self._backend.item(self._data)

    def __float__(self) -> float:
        return float(self.item())

    def __bool__(self) -> bool:
        return bool(self.item())

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("rank-0 Tensor has no length")
        return self._shape[0]

    def __getitem__(self, idx: int | slice | tuple[int | slice, ...]) -> Any:
        if not self._shape:
            raise TypeError("rank-0 Tensor is not subscriptable")
        if _has_slice(idx):
            raw = self._backend.slice_get(self._data, idx, self._shape)
            return Tensor._wrap(raw, self._backend)
        if isinstance(idx, tuple):
            result: Any = self
            for i in idx:
                if not isinstance(result, Tensor):
                    raise IndexError("too many indices for Tensor")
                result = result[i]
            return result
        item = self._data[idx]
        if len(self._shape) == 1:
            return item  # scalar (Python float or np.float64)
        return Tensor._wrap(item, self._backend)

    def __setitem__(
        self, idx: int | slice | tuple[int | slice, ...], value: Any
    ) -> None:
        if not self._shape:
            raise TypeError("rank-0 Tensor is not subscriptable")
        v = value._data if isinstance(value, Tensor) else value
        self._data = self._backend.slice_set(self._data, idx, v, self._shape)

    def __iter__(self) -> Iterator[Any]:
        if not self._shape:
            raise TypeError("rank-0 Tensor is not iterable")
        for i in range(self._shape[0]):
            yield self[i]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> Tensor:
        """Return a deep copy of this Tensor (same backend)."""
        return Tensor._wrap(self._backend.copy(self._data), self._backend)

    def to_list(self) -> Any:
        """Scalar for rank-0; deep-copied nested list for rank ≥ 1."""
        return self._backend.from_native(self._data)

    def __repr__(self) -> str:
        return f"Tensor({self._backend.from_native(self._data)!r})"

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
        return Tensor._wrap(self._backend.neg(self._data), self._backend)

    def __add__(self, other: Tensor) -> Tensor:
        self._check_backend(other)
        return Tensor._wrap(self._backend.add(self._data, other._data), self._backend)

    def __sub__(self, other: Tensor) -> Tensor:
        self._check_backend(other)
        return Tensor._wrap(self._backend.sub(self._data, other._data), self._backend)

    def __mul__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            self._check_backend(other)
            return Tensor._wrap(
                self._backend.mul_elem(self._data, other._data), self._backend
            )
        return Tensor._wrap(
            self._backend.mul_scalar(self._data, float(other)), self._backend
        )

    def __rmul__(self, other: Any) -> Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            self._check_backend(other)
            return Tensor._wrap(
                self._backend.div_elem(self._data, other._data), self._backend
            )
        return Tensor._wrap(
            self._backend.div_scalar(self._data, float(other)), self._backend
        )

    def __rtruediv__(self, other: float) -> Tensor:
        """Compute other / self element-wise (e.g. 2.0 / scalar_tensor)."""
        return Tensor._wrap(
            self._backend.rdiv_scalar(float(other), self._data), self._backend
        )

    def __lt__(self, other: Any) -> Tensor:
        other_raw = (
            other._data if isinstance(other, Tensor) else self._backend.to_native(other)
        )
        return Tensor._wrap(self._backend.lt(self._data, other_raw), self._backend)

    def __le__(self, other: Any) -> Tensor:
        other_raw = (
            other._data if isinstance(other, Tensor) else self._backend.to_native(other)
        )
        return Tensor._wrap(self._backend.le(self._data, other_raw), self._backend)

    def __gt__(self, other: Any) -> Tensor:
        other_raw = (
            other._data if isinstance(other, Tensor) else self._backend.to_native(other)
        )
        return Tensor._wrap(self._backend.gt(self._data, other_raw), self._backend)

    def __ge__(self, other: Any) -> Tensor:
        other_raw = (
            other._data if isinstance(other, Tensor) else self._backend.to_native(other)
        )
        return Tensor._wrap(self._backend.ge(self._data, other_raw), self._backend)

    def __matmul__(self, other: Tensor) -> Tensor:
        self._check_backend(other)
        raw = self._backend.matmul(self._data, other._data, self._shape, other._shape)
        return Tensor._wrap(raw, self._backend)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def abs(self) -> Tensor:
        """Element-wise absolute value."""
        return Tensor._wrap(self._backend.abs(self._data), self._backend)

    def max(self) -> Tensor:
        """Maximum of all elements; returns a 0-d scalar Tensor."""
        return Tensor._wrap(self._backend.reduce_max(self._data), self._backend)

    def argmax(self) -> Any:
        """Index of the maximum element in a 1-d Tensor."""
        return self._backend.argmax(self._data)

    def element(self, *indices: int) -> Tensor:
        """Return the scalar at the given static integer indices as a 0-d Tensor."""
        raw = self._data
        for i in indices:
            raw = raw[i]
        return Tensor._wrap(raw, self._backend)

    def take(self, indices: Tensor) -> Tensor:
        """Gather elements at integer indices; return a new Tensor."""
        self._check_backend(indices)
        return Tensor._wrap(
            self._backend.take(self._data, indices._data), self._backend
        )

    def diag(self) -> Tensor:
        """Main diagonal of a rank-2 Tensor → rank-1."""
        if len(self._shape) != 2:
            raise ValueError(f"diag requires rank-2 Tensor, got shape {self._shape}")
        return Tensor._wrap(self._backend.diag(self._data, self._shape), self._backend)

    def norm(self) -> float:
        """Frobenius norm (absolute value for rank-0)."""
        if not self._shape:
            return abs(float(self._data))
        return self._backend.norm(self._data)

    def svd(self) -> tuple[Tensor, Tensor, Tensor]:
        """One-sided Jacobi SVD: returns (U, s, Vt) with singular values descending.

        Returns
        -------
        U   : Tensor of shape (m, n) — left singular vectors as columns
        s   : Tensor of shape (n,)   — singular values, descending order
        Vt  : Tensor of shape (n, n) — right singular vectors as rows
        """
        if len(self._shape) != 2:
            raise ValueError(f"svd requires rank-2 Tensor, got shape {self._shape}")
        u_raw, s_raw, vt_raw = self._backend.svd(self._data, self._shape)
        return (
            Tensor._wrap(u_raw, self._backend),
            Tensor._wrap(s_raw, self._backend),
            Tensor._wrap(vt_raw, self._backend),
        )

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def zeros(cls, *shape: int, backend: Backend | None = None) -> Tensor:
        """Zero Tensor of the given shape (any rank, including rank-0)."""
        b = backend if backend is not None else get_default_backend()
        return cls._wrap(b.zeros(shape), b)

    @classmethod
    def eye(cls, n: int, *, backend: Backend | None = None) -> Tensor:
        """n × n identity matrix."""
        b = backend if backend is not None else get_default_backend()
        return cls._wrap(b.eye(n), b)


# ---------------------------------------------------------------------------
# Module-level einsum
# ---------------------------------------------------------------------------


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
    if not tensors:
        raise ValueError("einsum requires at least one tensor")
    backend = tensors[0]._backend
    for t in tensors[1:]:
        if t._backend is not backend:
            raise ValueError(
                f"all tensors in einsum must use the same backend; "
                f"got {type(backend).__name__} and {type(t._backend).__name__}"
            )
    raws = [t._data for t in tensors]
    shapes = [t._shape for t in tensors]
    return Tensor._wrap(backend.einsum(spec, raws, shapes), backend)


def where(cond: Tensor, x: Any, y: Any) -> Tensor:
    """Element-wise ternary: return x where cond is True, y elsewhere.

    cond, x, and y are broadcast against each other following the same
    rules as ``numpy.where``.  x and y may be Tensors or Python scalars.

    Parameters
    ----------
    cond:
        Boolean Tensor (from a comparison operator).
    x:
        Value(s) to use where cond is True.
    y:
        Value(s) to use where cond is False.
    """
    backend = cond._backend
    x_raw = x._data if isinstance(x, Tensor) else backend.to_native(x)
    y_raw = y._data if isinstance(y, Tensor) else backend.to_native(y)
    return Tensor._wrap(backend.where(cond._data, x_raw, y_raw), backend)


def arange(n: int, *, backend: Backend | None = None) -> Tensor:
    """Return a 1-d integer Tensor [0, 1, ..., n-1].

    Parameters
    ----------
    n:
        Length of the output Tensor.
    backend:
        Backend to use; defaults to the process-wide default.
    """
    b = backend if backend is not None else get_default_backend()
    return Tensor._wrap(b.arange(n), b)


__all__ = ["Real", "Tensor", "arange", "einsum", "where"]
