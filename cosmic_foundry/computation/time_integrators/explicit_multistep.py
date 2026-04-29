"""Explicit linear multistep integrators (Adams-Bashforth family).

An Adams-Bashforth k-step method advances the ODE y' = f(t, y) using the k
most recent function evaluations:

    y_{n+1} = y_n + h · Σ_{j=0}^{k-1} β_j · f_{n-j}

The coefficients β_j are the classical Adams-Bashforth quadrature weights;
they satisfy the LMM order conditions of order k.  The first k−1 steps are
bootstrapped with RK4 to populate the history without degrading accuracy.

MultistepState extends the four-field (t, u, dt, err) step state with a
history field that carries past function evaluations.  An initial state with
empty history triggers bootstrapping; each bootstrap call appends one entry
until the history is full and the full AB formula takes over.

Named instances
---------------
ab2  — order 2, β = [3/2, −1/2]
ab3  — order 3, β = [23/12, −16/12, 5/12]
ab4  — order 4, β = [55/24, −59/24, 37/24, −9/24]
"""

from __future__ import annotations

from typing import NamedTuple

import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.integrator import (
    RHSProtocol,
    RKState,
    TimeIntegrator,
)
from cosmic_foundry.computation.time_integrators.runge_kutta import rk4 as _rk4


class MultistepState(NamedTuple):
    """State for explicit linear multistep integrators.

    Extends the four-field step state with a history tuple of past function
    evaluations.  Single-step integrators never read or write the fifth
    field; multistep integrators update it on every step.

    Fields
    ------
    t:
        Current time.
    u:
        Current solution as a Tensor.
    dt:
        Step size used to arrive at this state (0.0 before any step).
    err:
        Embedded error estimate norm (0.0 for methods without embedded pairs).
    history:
        Tuple of previous function evaluations (f_{n-1}, f_{n-2}, …),
        most recent first.  Empty for the initial state; filled by bootstrap.
    """

    t: float
    u: Tensor
    dt: float = 0.0
    err: float = 0.0
    history: tuple[Tensor, ...] = ()


class ExplicitMultistepIntegrator(TimeIntegrator):
    """Explicit Adams-Bashforth k-step method parameterized by quadrature weights.

    The first k−1 steps are bootstrapped with RK4 to build the required
    function-value history; all subsequent steps use the full AB formula.

    step() accepts an RKState (as seeded by TimeStepper on the first call)
    or a MultistepState returned by a previous step.  It always returns
    MultistepState.

    Parameters
    ----------
    beta:
        Adams-Bashforth quadrature weights [β₀, β₁, …, β_{k-1}], where β₀
        multiplies f_n (current) and β_{k-1} multiplies f_{n-k+1} (oldest).
        Accepted as sympy-compatible values; stored as floats for evaluation.
    order:
        Declared convergence order.
    """

    def __init__(self, beta: list, order: int) -> None:
        self._beta_f = [float(sympy.sympify(b)) for b in beta]
        self._order = order
        self._k = len(beta)

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._order

    def step(
        self,
        rhs: RHSProtocol,
        state: RKState | MultistepState,
        dt: float,
    ) -> MultistepState:
        """Advance state by one step of size dt.

        Uses one RK4 step for each of the first k−1 bootstrap calls, then
        switches to the full Adams-Bashforth formula once the history is full.
        """
        t, u = state.t, state.u
        history = state.history if isinstance(state, MultistepState) else ()

        if len(history) < self._k - 1:
            f_n = rhs(t, u)
            rk4_state: RKState = _rk4.step(rhs, RKState(t, u), dt)
            return MultistepState(rk4_state.t, rk4_state.u, dt, 0.0, (f_n,) + history)

        f_n = rhs(t, u)
        all_f = (f_n,) + history[: self._k - 1]
        u_new = u
        for beta_j, f_j in zip(self._beta_f, all_f, strict=True):
            u_new = u_new + (beta_j * dt) * f_j
        return MultistepState(t + dt, u_new, dt, 0.0, all_f[:-1])


# ---------------------------------------------------------------------------
# Named instances
# ---------------------------------------------------------------------------

ab2 = ExplicitMultistepIntegrator(
    beta=[sympy.Rational(3, 2), sympy.Rational(-1, 2)],
    order=2,
)

ab3 = ExplicitMultistepIntegrator(
    beta=[sympy.Rational(23, 12), sympy.Rational(-16, 12), sympy.Rational(5, 12)],
    order=3,
)

ab4 = ExplicitMultistepIntegrator(
    beta=[
        sympy.Rational(55, 24),
        sympy.Rational(-59, 24),
        sympy.Rational(37, 24),
        sympy.Rational(-9, 24),
    ],
    order=4,
)


__all__ = [
    "ExplicitMultistepIntegrator",
    "MultistepState",
    "ab2",
    "ab3",
    "ab4",
]
