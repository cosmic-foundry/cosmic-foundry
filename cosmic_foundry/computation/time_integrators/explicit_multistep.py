"""Explicit linear multistep integrators (Adams-Bashforth family).

An Adams-Bashforth k-step method advances the ODE y' = f(t, y) using the k
most recent function evaluations:

    y_{n+1} = y_n + h · Σ_{j=0}^{k-1} β_j · f_{n-j}

The coefficients β_j are the classical Adams-Bashforth quadrature weights;
they satisfy the LMM order conditions of order k.  The first k−1 steps are
bootstrapped with RK4 to populate the history without degrading accuracy.

The initial state is an ODEState with history=None; each bootstrap call
populates history until it holds k−1 entries and the full AB formula takes
over.  Past function evaluations are stored most-recent-first in
ODEState.history.

Named instances
---------------
ab1  — order 1, β = [1]
ab2  — order 2, β = [3/2, −1/2]
ab3  — order 3, β = [23/12, −16/12, 5/12]
ab4  — order 4, β = [55/24, −59/24, 37/24, −9/24]
ab5  — order 5, β = [1901/720, −1387/360, 109/30, −637/360, 251/720]
ab6  — order 6, β = [4277/1440, −2641/480, 4991/720, −3649/720, 959/480, −95/288]
"""

from __future__ import annotations

import sympy

from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    RHSProtocol,
    TimeIntegrator,
)
from cosmic_foundry.computation.time_integrators.runge_kutta import (
    RungeKuttaIntegrator as _RungeKuttaIntegrator,
)

_bootstrap_rk = _RungeKuttaIntegrator(6)


class ExplicitMultistepIntegrator(TimeIntegrator):
    """Explicit Adams-Bashforth k-step method parameterized by quadrature weights.

    The first k−1 steps are bootstrapped with RK4 to build the required
    function-value history; all subsequent steps use the full AB formula.

    step() accepts an ODEState with history=None (as seeded by Integrator on
    the first call) or an ODEState with history populated by a previous step.
    It always returns ODEState.

    Parameters
    ----------
    beta:
        Adams-Bashforth quadrature weights [β₀, β₁, …, β_{k-1}], where β₀
        multiplies f_n (current) and β_{k-1} multiplies f_{n-k+1} (oldest).
        Accepted as sympy-compatible values; stored as floats for evaluation.
    order:
        Declared convergence order.
    """

    # Classical Adams-Bashforth quadrature weights for orders 1–6.
    # Keys are order; values are (beta_list, order) ready for __init__.
    _AB: dict[int, list] = {
        1: [sympy.Rational(1, 1)],
        2: [sympy.Rational(3, 2), sympy.Rational(-1, 2)],
        3: [sympy.Rational(23, 12), sympy.Rational(-16, 12), sympy.Rational(5, 12)],
        4: [
            sympy.Rational(55, 24),
            sympy.Rational(-59, 24),
            sympy.Rational(37, 24),
            sympy.Rational(-9, 24),
        ],
        5: [
            sympy.Rational(1901, 720),
            sympy.Rational(-1387, 360),
            sympy.Rational(109, 30),
            sympy.Rational(-637, 360),
            sympy.Rational(251, 720),
        ],
        6: [
            sympy.Rational(4277, 1440),
            sympy.Rational(-2641, 480),
            sympy.Rational(4991, 720),
            sympy.Rational(-3649, 720),
            sympy.Rational(959, 480),
            sympy.Rational(-95, 288),
        ],
    }

    @classmethod
    def for_order(cls, q: int) -> ExplicitMultistepIntegrator:
        """Return the standard Adams-Bashforth integrator of order q (1–6)."""
        if q not in cls._AB:
            raise ValueError(
                f"Adams-Bashforth order {q} not available; "
                f"choose from {sorted(cls._AB)}"
            )
        return cls(cls._AB[q], q)

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
        state: ODEState,
        dt: float,
    ) -> ODEState:
        """Advance state by one step of size dt.

        Uses one RK4 step for each of the first k−1 bootstrap calls, then
        switches to the full Adams-Bashforth formula once the history is full.
        """
        t, u = state.t, state.u
        history = () if state.history is None else state.history

        if len(history) < self._k - 1:
            f_n = rhs(t, u)
            boot_state: ODEState = _bootstrap_rk.step(rhs, ODEState(t, u), dt)
            return ODEState(boot_state.t, boot_state.u, dt, 0.0, (f_n,) + history)

        f_n = rhs(t, u)
        all_f = (f_n,) + history[: self._k - 1]
        u_new = u
        for beta_j, f_j in zip(self._beta_f, all_f, strict=True):
            u_new = u_new + (beta_j * dt) * f_j
        return ODEState(t + dt, u_new, dt, 0.0, all_f[:-1])


__all__ = [
    "ExplicitMultistepIntegrator",
]
