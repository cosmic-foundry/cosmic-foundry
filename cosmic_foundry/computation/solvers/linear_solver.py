"""LinearSolver ABC and LinearOperator protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from cosmic_foundry.computation.tensor import Tensor


class LinearOperator(Protocol):
    """Structural protocol satisfied by any JIT-compilable linear operator.

    Operator in physics/ satisfies this without any import relationship:
    computation/ never imports from physics/.  Solvers depend only on this
    protocol, not on any concrete operator class.
    """

    def apply(self, u: Tensor) -> Tensor:
        """Apply the operator to u; return result of the same shape."""
        ...

    def diagonal(self, backend: Any) -> Tensor:
        """Return the diagonal entries A[i,i] as a 1-D Tensor."""
        ...

    def row_abs_sums(self, backend: Any) -> Tensor:
        """Return per-row sums of |A[i,j]| as a 1-D Tensor."""
        ...


class LinearSolver(ABC):
    """Abstract interface for solving a linear system A u = b.

    Solvers receive a LinearOperator (not a pre-assembled matrix) and derive
    whatever matrix representation they need internally.  Iterative solvers
    use op.apply() directly; direct solvers assemble the matrix from N
    op.apply() calls on basis vectors.

    The empirical cost scaling exponent p (such that T_solve(N) ≈ α · N^p) is
    measured by the Autotuner rather than declared here.
    """

    @abstractmethod
    def solve(self, op: LinearOperator, b: Tensor) -> Tensor:
        """Solve A u = b for u; return the solution Tensor.

        Parameters
        ----------
        op:
            Linear operator providing apply(), diagonal(), and row_abs_sums().
        b:
            Right-hand-side vector, shape (n,).

        Returns
        -------
        Tensor
            u of shape (n,) satisfying ‖A u − b‖₂ < solver tolerance.
        """


class TensorLinearOperator:
    """Wraps a pre-assembled Tensor matrix as a LinearOperator.

    Used by the benchmarker and tests that have a matrix already in hand.
    apply() delegates to matrix-vector multiply; diagonal() and row_abs_sums()
    are extracted directly from the matrix entries.
    """

    def __init__(self, a: Tensor) -> None:
        self._a = a

    def apply(self, u: Tensor) -> Tensor:
        return self._a @ u

    def diagonal(self, backend: Any) -> Tensor:
        from cosmic_foundry.computation import tensor

        return tensor.diag(self._a)

    def row_abs_sums(self, backend: Any) -> Tensor:
        from cosmic_foundry.computation import tensor

        return tensor.einsum("ij->i", tensor.abs(self._a))


__all__ = ["LinearOperator", "LinearSolver", "TensorLinearOperator"]
