"""IterativeSolver: LinearSolver driven by a step/converged loop."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from cosmic_foundry.computation.linear_solver import LinearSolver
from cosmic_foundry.computation.tensor import Tensor


class IterativeSolver(LinearSolver):
    """LinearSolver that advances a state via repeated step() calls.

    Subclasses implement init_state, step, converged, and extract.
    The base solve() runs the Python while loop.

    The step/converged/state decomposition is designed to map onto
    jax.lax.while_loop: state is a pytree of arrays, step is a pure
    function, and converged returns a boolean.  Subclasses must ensure
    their state contains only Tensors, scalars, and nested tuples of
    these so that JAX can trace the loop body.

    Required:
        init_state — construct initial state from (a, b)
        step       — advance state by one iteration; return new state
        converged  — return True when the iteration should stop
        extract    — extract the solution Tensor from the final state
    """

    @abstractmethod
    def init_state(self, a: Tensor, b: Tensor) -> Any:
        """Construct the initial solver state from the system (a, b)."""

    @abstractmethod
    def step(self, state: Any) -> Any:
        """Advance the solver state by one iteration; return new state."""

    @abstractmethod
    def converged(self, state: Any) -> Tensor:
        """Return a 0-d bool Tensor: True when converged or max_iter reached."""

    @abstractmethod
    def extract(self, state: Any) -> Tensor:
        """Extract the solution Tensor from the final state."""

    def solve(self, a: Tensor, b: Tensor) -> Tensor:
        state = self.init_state(a, b)
        while not self.converged(state).item():
            state = self.step(state)
        return self.extract(state)


__all__ = ["IterativeSolver"]
