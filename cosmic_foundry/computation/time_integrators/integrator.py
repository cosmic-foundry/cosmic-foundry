"""TimeIntegrator ABC plus the Phase-0 typed slot inhabitants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RHSProtocol(Protocol):
    """Protocol for ODE right-hand sides du/dt = f(t, u).

    f : ℝ × ℝⁿ → ℝⁿ.  Every concrete RHS that integrators accept must
    satisfy this protocol.  Phase 0 provides BlackBoxRHS; later phases add
    structured variants (WithJacobianRHS, AdditiveRHS, HamiltonianSplit).

    The state vector u is a plain numpy ndarray so that the protocol is
    backend-agnostic at the ABC level.  Integrators that target JAX arrays
    accept RHSProtocol and document the array type in their own docstring.
    """

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        """Evaluate du/dt at time t given state vector u."""
        ...


class BlackBoxRHS:
    """RHSProtocol wrapping an arbitrary callable f(t, u).

    The canonical Phase-0 inhabitant of the RHS slot.  No structure is
    assumed beyond the f(t, u) → u shape; later phases replace this with
    richer protocols that expose Jacobians, linear/nonlinear splitting, or
    Hamiltonian structure.

    Parameters
    ----------
    f:
        Callable with signature (t: float, u: ndarray) → ndarray.
    """

    def __init__(self, f: Any) -> None:
        self._f = f

    def __call__(self, t: float, u: np.ndarray) -> np.ndarray:
        return np.asarray(self._f(t, u))


class RKState(NamedTuple):
    """Integrator state for Runge-Kutta methods.

    Carries the minimal information the stepper needs after each accepted step.
    The typed state slot is DSL-ready: Phase 1 replaces this with a richer
    state that also carries dt_suggest and the embedded error estimate.

    Fields
    ------
    t:
        Current time.
    u:
        Current state vector, shape (n,).
    """

    t: float
    u: np.ndarray


class Controller(Protocol):
    """Protocol for step-size controllers.

    A controller maps the current state to the next step size.  Phase 0
    provides ConstantStep; Phase 1 adds PIController with embedded error
    estimation.

    suggest(state) → dt
        Returns the step size to use for the step starting at state.t.
    """

    def suggest(self, state: RKState) -> float:
        """Return the step size to use for the step starting at state.t."""
        ...


class ConstantStep:
    """Controller that returns a fixed step size on every call.

    The canonical Phase-0 inhabitant of the Controller slot.  No error
    estimation or rejection logic; the step size is set at construction
    and never changed.

    Parameters
    ----------
    dt:
        Step size to return on every call to suggest().
    """

    def __init__(self, dt: float) -> None:
        self._dt = dt

    def suggest(self, state: RKState) -> float:
        return self._dt


class TimeIntegrator(ABC):
    """Abstract base for all time integrators.

    Defines the interface shared by every Phase-0–9 integrator.  The three
    typed slots — rhs (RHSProtocol), state type (RKState for Phase 0), and
    controller (Controller) — are DSL-ready: each subsequent phase replaces
    exactly one slot with a richer variant without changing this interface.

    Subclasses implement step(), which advances the integrator state by one
    step of size dt.  The stepper's job is to call step() repeatedly via the
    controller's suggested dt.

    Required:
        order — declared convergence order
        step  — advance state by one step of size dt
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """Declared convergence order of the method."""

    @abstractmethod
    def step(self, rhs: RHSProtocol, state: RKState, dt: float) -> RKState:
        """Advance state by one step of size dt.

        Returns a new RKState with t = state.t + dt and updated u.
        """


__all__ = [
    "BlackBoxRHS",
    "ConstantStep",
    "Controller",
    "RHSProtocol",
    "RKState",
    "TimeIntegrator",
]
