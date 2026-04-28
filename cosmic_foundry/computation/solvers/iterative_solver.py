"""IterativeSolver: LinearSolver driven by a step/converged loop."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from cosmic_foundry.computation.solvers.linear_solver import (
    LinearOperator,
    LinearSolver,
)
from cosmic_foundry.computation.tensor import Tensor


class IterativeSolver(LinearSolver):
    """LinearSolver that advances a state via repeated step() calls.

    Subclasses implement init_state, step, converged, and extract.
    The base solve() runs the while loop via the backend, passing op to each
    step via closure so that the while_loop body has the fixed signature
    state -> state required by JAX's lax.while_loop.

    The step/converged/state decomposition is designed to map onto
    jax.lax.while_loop: state is a pytree of arrays, step is a pure
    function, and converged returns a boolean.  Subclasses must ensure
    their state contains only Tensors, scalars, and nested tuples of
    these so that JAX can trace the loop body.

    Required:
        init_state — construct initial state from (op, b)
        step       — advance state by one iteration given (op, state)
        converged  — return True when the iteration should stop
        extract    — extract the solution Tensor from the final state
    """

    @abstractmethod
    def init_state(self, op: LinearOperator, b: Tensor) -> Any:
        """Construct the initial solver state from the operator and RHS."""

    @abstractmethod
    def step(self, op: LinearOperator, state: Any) -> Any:
        """Advance the solver state by one iteration; return new state."""

    @abstractmethod
    def converged(self, state: Any) -> Tensor:
        """Return a 0-d bool Tensor: True when converged or max_iter reached."""

    @abstractmethod
    def extract(self, state: Any) -> Tensor:
        """Extract the solution Tensor from the final state."""

    def solve(self, op: LinearOperator, b: Tensor) -> Tensor:
        init = self.init_state(op, b)
        final = b.backend.while_loop(
            cond_fn=lambda s: ~self.converged(s),
            body_fn=lambda s: self.step(op, s),
            init_state=init,
        )
        return self.extract(final)


__all__ = ["IterativeSolver"]
