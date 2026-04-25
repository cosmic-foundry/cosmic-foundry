"""NumPy backend for Tensor: all operations delegate to numpy."""

from __future__ import annotations

from typing import Any

import numpy as np


class NumpyBackend:
    """Implements all Tensor operations using NumPy arrays.

    All raw values are ``np.ndarray``.  Rank-0 tensors are stored as
    0-dimensional arrays so that shape inference is uniform.

    Parameters
    ----------
    dtype:
        Element dtype for ``zeros`` and ``eye`` and for coercing input in
        ``to_native``.  When ``None`` (the default) the dtype is inferred
        from the input data, so ``float``/``int`` inputs stay at NumPy's
        natural promotion (typically float64) while existing ``np.float32``
        arrays are preserved as-is.
    """

    def __init__(self, dtype: np.dtype | type | None = None) -> None:
        self._dtype = dtype

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def to_native(self, data: Any) -> np.ndarray:
        """Convert Python scalar or nested sequence to an ndarray."""
        return np.asarray(data, dtype=self._dtype)

    def from_native(self, raw: np.ndarray) -> Any:
        """Convert ndarray to a Python scalar or nested list."""
        if raw.ndim == 0:
            return float(raw)
        return raw.tolist()

    def infer_shape(self, raw: np.ndarray) -> tuple[int, ...]:
        return tuple(raw.shape)

    def copy(self, raw: np.ndarray) -> np.ndarray:
        return raw.copy()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def zeros(self, shape: tuple[int, ...]) -> np.ndarray:
        dt = self._dtype if self._dtype is not None else np.float64
        if not shape:
            return np.array(0.0, dtype=dt)
        return np.zeros(shape, dtype=dt)

    def eye(self, n: int) -> np.ndarray:
        dt = self._dtype if self._dtype is not None else np.float64
        return np.eye(n, dtype=dt)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def flatten(self, raw: np.ndarray) -> list[float]:
        return raw.ravel().tolist()

    def norm(self, raw: np.ndarray) -> float:
        return float(np.linalg.norm(raw))

    # ------------------------------------------------------------------
    # Element-wise arithmetic
    # ------------------------------------------------------------------

    def neg(self, a: np.ndarray) -> Any:
        return np.negative(a)

    def add(self, a: np.ndarray, b: np.ndarray) -> Any:
        return np.add(a, b)

    def sub(self, a: np.ndarray, b: np.ndarray) -> Any:
        return np.subtract(a, b)

    def mul_scalar(self, a: np.ndarray, s: float) -> Any:
        return np.multiply(a, s)

    def mul_elem(self, a: np.ndarray, b: np.ndarray) -> Any:
        return np.multiply(a, b)

    def div_scalar(self, a: np.ndarray, s: float) -> Any:
        return np.divide(a, s)

    def div_elem(self, a: np.ndarray, b: np.ndarray) -> Any:
        return np.divide(a, b)

    # ------------------------------------------------------------------
    # Contraction
    # ------------------------------------------------------------------

    def matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        shape_a: tuple[int, ...],
        shape_b: tuple[int, ...],
    ) -> Any:
        return np.matmul(a, b)

    def einsum(
        self,
        spec: str,
        raws: list[np.ndarray],
        shapes: list[tuple[int, ...]],
    ) -> Any:
        return np.einsum(spec, *raws)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def diag(self, raw: np.ndarray, shape: tuple[int, ...]) -> Any:
        return np.diag(raw)

    def svd(
        self, raw: np.ndarray, shape: tuple[int, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, s, Vt = np.linalg.svd(raw, full_matrices=False)
        return U, s, Vt
