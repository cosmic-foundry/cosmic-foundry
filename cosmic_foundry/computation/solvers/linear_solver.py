"""LinearSolver ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cosmic_foundry.computation.tensor import Tensor


class LinearSolver(ABC):
    """Abstract interface for solving a linear system A u = b.

    A LinearSolver accepts an assembled stiffness matrix A and right-hand-side
    vector b as Tensors and returns the solution vector u as a Tensor.  It is
    deliberately mesh-agnostic: assembly and index mapping are the caller's
    responsibility.  This separation lets solvers be swapped without touching
    discretization code, and lets the same solver be reused across time steps
    when A is constant.

    The empirical cost scaling exponent p (such that T_solve(N) ≈ α · N^p) is
    measured by the Autotuner rather than declared here; theoretical values will
    be derived by symbolic op-tracing in a future pass.
    """

    @abstractmethod
    def solve(self, a: Tensor, b: Tensor) -> Tensor:
        """Solve A u = b for u; return the solution Tensor.

        Parameters
        ----------
        a:
            Assembled stiffness matrix, shape (n, n).
        b:
            Right-hand-side vector, shape (n,).

        Returns
        -------
        Tensor
            u of shape (n,) satisfying ‖A u − b‖₂ < solver tolerance.
        """


__all__ = ["LinearSolver"]
