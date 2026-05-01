"""TimeIntegrator ABC plus the Phase-0 and Phase-1 typed slot inhabitants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Protocol, runtime_checkable

from cosmic_foundry.computation.tensor import Tensor


@runtime_checkable
class RHSProtocol(Protocol):
    """Protocol for ODE right-hand sides du/dt = f(t, u).

    f : ℝ × ℝⁿ → ℝⁿ.  Every concrete RHS that integrators accept must
    satisfy this protocol.  Phase 0 provides BlackBoxRHS; later phases add
    structured variants (WithJacobianRHS, SplitRHS, HamiltonianRHS).
    """

    def __call__(self, t: float, u: Tensor) -> Tensor:
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
        Callable with signature (t: float, u: Tensor) → Tensor.
    """

    def __init__(self, f: Any) -> None:
        self._f = f

    def __call__(self, t: float, u: Tensor) -> Tensor:
        result: Tensor = self._f(t, u)
        return result


class ODEState(NamedTuple):
    """Integrator state for ODE solvers.

    Carries all information the stepper and controller need after each
    attempted step.

    Fields
    ------
    t:
        Current time.
    u:
        Current state vector as a Tensor.
    dt:
        Step size used to arrive at this state (0.0 for the initial state
        before any step is taken).
    err:
        Raw L2 error norm from the embedded estimate for the step that
        produced this state (0.0 if the integrator has no embedded pair or
        for the initial state).
    history:
        Tuple of past function evaluations for multistep integrators
        (most recent first), or None for single-step methods.
    active_constraints:
        Frozenset of reaction-pair indices currently treated as algebraic
        constraints.  ``None`` (the default) means no constraint tracking.
        A frozenset means those pairs are algebraic constraints; the
        ``ConstraintAwareController`` manages this field across steps.
        Integrators pass it through without interpreting it.
    """

    t: float
    u: Tensor
    dt: float = 0.0
    err: float = 0.0
    history: Any = None
    active_constraints: frozenset[int] | None = None


class Controller(Protocol):
    """Protocol for step-size controllers.

    A controller decides whether to accept a step and suggests the next
    step size.  Phase 0 provides ConstantStep; Phase 1 adds PIController.

    Methods
    -------
    accept(state) → bool
        Return True when the step that produced state should be accepted.
        Always True for non-adaptive controllers.
    suggest(state, *, accepted) → float
        Return the step size for the next (or retry) step.  `accepted`
        mirrors the result of accept() and tells the controller whether
        to update its internal memory (e.g. the previous error in a PI
        controller).
    """

    def accept(self, state: ODEState) -> bool:
        """Return True iff the step that produced state is accepted."""
        ...

    def suggest(self, state: ODEState, *, accepted: bool = True) -> float:
        """Return the step size to use next (or to retry with on rejection)."""
        ...


class ConstantStep:
    """Controller that returns a fixed step size on every call.

    The canonical Phase-0 inhabitant of the Controller slot.  No error
    estimation or rejection logic; every step is accepted and the step
    size is set at construction and never changed.

    Parameters
    ----------
    dt:
        Step size to return on every call to suggest().
    """

    def __init__(self, dt: float) -> None:
        self._dt = dt

    def accept(self, state: ODEState) -> bool:
        return True

    def suggest(self, state: ODEState, *, accepted: bool = True) -> float:
        return self._dt


class PIController:
    """Proportional-integral step-size controller over an embedded error estimate.

    Implements the Gustafsson PI formula:

        h_{n+1} = h_n · safety · (tol / err_n)^α · (tol / err_{n-1})^β

    where err_n is the L2 norm of the difference between the main and
    embedded solutions, and err_{n-1} is the last accepted error (or tol
    on the first step, giving a neutral factor of 1).  The factor is
    clamped to [factor_min, factor_max] before scaling h_n.

    The controller is stateful: it tracks the last accepted error and
    updates it only on acceptance, so that rejected steps do not corrupt
    the integral memory.

    Typical exponents (Hairer Vol. II): α = 0.7/p, β = 0.4/p for a
    p-th order method.  A pure I-controller uses β = 0.

    Parameters
    ----------
    alpha:
        Proportional exponent.
    beta:
        Integral exponent.
    tol:
        Acceptance threshold: a step is accepted when err ≤ tol.
    dt0:
        Initial step size, returned by suggest() before any step is taken.
    safety:
        Safety factor multiplied into the PI factor.  Default 0.9.
    factor_min:
        Minimum allowed step-size growth factor.  Default 0.1.
    factor_max:
        Maximum allowed step-size growth factor.  Default 10.0.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        tol: float,
        dt0: float,
        safety: float = 0.9,
        factor_min: float = 0.1,
        factor_max: float = 10.0,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._tol = tol
        self._dt0 = dt0
        self._safety = safety
        self._factor_min = factor_min
        self._factor_max = factor_max
        self._err_prev = tol  # neutral: tol/err_prev = 1 on first step

    @property
    def factor_min(self) -> float:
        """Minimum permitted shrink factor for rejected steps."""
        return self._factor_min

    @property
    def factor_max(self) -> float:
        """Maximum permitted growth factor for the next suggested step."""
        return self._factor_max

    def accept(self, state: ODEState) -> bool:
        """Accept when err ≤ tol or when there is no embedded estimate."""
        return state.err == 0.0 or state.err <= self._tol

    def suggest(self, state: ODEState, *, accepted: bool = True) -> float:
        """Compute next step size via the PI formula.

        On acceptance: full PI formula using current error and the previous
        accepted error; err_prev is updated after the factor is computed.
        On rejection: pure proportional term only, so that the integral
        memory from a previously very successful step cannot push the retry
        step size upward instead of downward.
        When no embedded estimate is available (state.err == 0 or
        state.dt == 0), returns dt0.
        """
        if state.err == 0.0 or state.dt == 0.0:
            return self._dt0
        err = state.err
        factor: float
        if accepted:
            factor = self._safety * (
                (self._tol / err) ** self._alpha
                * (self._tol / self._err_prev) ** self._beta
            )
            self._err_prev = err
        else:
            factor = self._safety * (self._tol / err) ** (self._alpha + self._beta)
        factor = max(self._factor_min, min(self._factor_max, factor))
        return state.dt * factor


class TimeIntegrator(ABC):
    """Abstract base for all time integrators.

    Each integrator accepts an RHS and a state object, advances one step of
    size dt, and returns a new state.  All integrators use ODEState as the
    unified state type.

    Required:
        order — declared convergence order
        step  — advance state by one step of size dt
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """Declared convergence order of the method."""

    @abstractmethod
    def step(self, rhs: Any, state: Any, dt: float) -> Any:
        """Advance state by one step of size dt.

        Returns a new state with t = state.t + dt, updated u, dt set to the
        step size used, and err set to the embedded error norm (0.0 if no
        embedded pair is available).  The concrete return type depends on the
        integrator family; all state types carry at least (t, u, dt, err).
        """


__all__ = [
    "BlackBoxRHS",
    "ConstantStep",
    "Controller",
    "ODEState",
    "PIController",
    "RHSProtocol",
    "TimeIntegrator",
]
