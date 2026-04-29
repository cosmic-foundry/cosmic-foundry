"""Explicit Adams-Bashforth linear multistep integrators.

An Adams-Bashforth k-step method advances the ODE y' = f(t, y) using the k
most recent function evaluations:

    y_{n+1} = y_n + h · Σ_{j=0}^{k-1} β_j · f_{n-j}

The coefficients β_j are the classical Adams-Bashforth quadrature weights;
they satisfy the LMM order conditions of order k.  The first k−1 steps are
bootstrapped with RK4 to populate the history without degrading accuracy.

ABState carries the current time, current solution, and the function-value
history (f_{n-1}, f_{n-2}, …, f_{n-k+1}) as a tuple, most recent first.
An initial ABState(t₀, u₀) has empty history; the bootstrap fills it step
by step.  Conservation laws preserved by the full RHS are also preserved by
every AB step: since Σ_i f_i(t, y) = 0 for a closed reaction network, the
same identity holds for any linear combination of past f values.

Named instances
---------------
ab2  — order 2, β = [3/2, −1/2]
ab3  — order 3, β = [23/12, −16/12, 5/12]
ab4  — order 4, β = [55/24, −59/24, 37/24, −9/24]
"""

from __future__ import annotations

import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.integrator import RHSProtocol, RKState
from cosmic_foundry.computation.time_integrators.runge_kutta import rk4 as _rk4


class ABState:
    """State for Adams-Bashforth multistep integrators.

    Parameters
    ----------
    t:
        Current time.
    u:
        Current solution as a Tensor.
    history:
        Tuple of previous function evaluations (f_{n-1}, f_{n-2}, …),
        most recent first.  Empty for the initial state; filled by bootstrap.
    """

    __slots__ = ("t", "u", "history")

    def __init__(
        self,
        t: float,
        u: Tensor,
        history: tuple[Tensor, ...] = (),
    ) -> None:
        self.t = t
        self.u = u
        self.history = history


class AdamsBashforthIntegrator:
    """Explicit Adams-Bashforth k-step method parameterized by quadrature weights.

    The first k−1 steps are bootstrapped with RK4 to build the required
    function-value history; all subsequent steps use the full AB formula.

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
        state: ABState,
        dt: float,
    ) -> ABState:
        """Advance state by one step of size dt.

        Uses one RK4 step for each of the first k−1 bootstrap calls, then
        switches to the full Adams-Bashforth formula once the history is full.
        """
        t, u = state.t, state.u
        history = state.history

        if len(history) < self._k - 1:
            # Bootstrap: take one RK4 step; store f_n for future AB use.
            f_n = rhs(t, u)
            rk4_state: RKState = _rk4.step(rhs, RKState(t, u), dt)
            return ABState(rk4_state.t, rk4_state.u, (f_n,) + history)

        # Full Adams-Bashforth step.
        f_n = rhs(t, u)
        all_f = (f_n,) + history[: self._k - 1]
        u_new = u
        for beta_j, f_j in zip(self._beta_f, all_f, strict=True):
            u_new = u_new + (beta_j * dt) * f_j
        return ABState(t + dt, u_new, all_f[:-1])


# ---------------------------------------------------------------------------
# Named instances
# ---------------------------------------------------------------------------

ab2 = AdamsBashforthIntegrator(
    beta=[sympy.Rational(3, 2), sympy.Rational(-1, 2)],
    order=2,
)

ab3 = AdamsBashforthIntegrator(
    beta=[sympy.Rational(23, 12), sympy.Rational(-16, 12), sympy.Rational(5, 12)],
    order=3,
)

ab4 = AdamsBashforthIntegrator(
    beta=[
        sympy.Rational(55, 24),
        sympy.Rational(-59, 24),
        sympy.Rational(37, 24),
        sympy.Rational(-9, 24),
    ],
    order=4,
)


__all__ = [
    "ABState",
    "AdamsBashforthIntegrator",
    "ab2",
    "ab3",
    "ab4",
]
