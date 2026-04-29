"""Explicit Runge-Kutta integrators parameterized by Butcher tableaux."""

from __future__ import annotations

import numpy as np
import sympy

from cosmic_foundry.computation.time_integrators.integrator import (
    RHSProtocol,
    RKState,
    TimeIntegrator,
)


class RungeKuttaIntegrator(TimeIntegrator):
    """Explicit Runge-Kutta method defined by a Butcher tableau (A, b, c, order).

    Given du/dt = f(t, u), advances u from tₙ to tₙ₊₁ = tₙ + h via:

        kᵢ = f(tₙ + cᵢ h, uₙ + h Σⱼ<ᵢ Aᵢⱼ kⱼ)   i = 1, …, s
        uₙ₊₁ = uₙ + h Σᵢ bᵢ kᵢ

    where A is strictly lower-triangular (explicit method), b are the weights,
    and c are the abscissae.  The declared order p is not verified by this
    class; verification is the responsibility of the test framework in
    tests/test_time_integrators.py via symbolic order-condition checks.

    Coefficients may be supplied as Python numbers, numpy arrays, or sympy
    Rationals.  Internally they are stored as sympy objects (for the order
    check framework) and evaluated to float64 numpy arrays for numerical use.

    Parameters
    ----------
    A:
        Stage interaction matrix, shape (s, s), strictly lower-triangular.
    b:
        Quadrature weights, shape (s,).  Must sum to 1.
    c:
        Abscissae (time nodes), shape (s,).  c[0] is conventionally 0.
    order:
        Declared convergence order of the method.
    """

    def __init__(
        self,
        A: list[list],
        b: list,
        c: list,
        order: int,
    ) -> None:
        self._A_sym = [[sympy.Rational(a) for a in row] for row in A]
        self._b_sym = [sympy.Rational(bi) for bi in b]
        self._c_sym = [sympy.Rational(ci) for ci in c]
        self._order = order

        s = len(b)
        self._A = np.array([[float(a) for a in row] for row in self._A_sym])
        self._b = np.array([float(bi) for bi in self._b_sym])
        self._c = np.array([float(ci) for ci in self._c_sym])
        self._s = s

    @property
    def order(self) -> int:
        return self._order

    @property
    def A_sym(self) -> list[list[sympy.Rational]]:
        """Butcher A matrix with sympy.Rational entries (for order verification)."""
        return self._A_sym

    @property
    def b_sym(self) -> list[sympy.Rational]:
        """Quadrature weights as sympy.Rational (for order verification)."""
        return self._b_sym

    @property
    def c_sym(self) -> list[sympy.Rational]:
        """Abscissae as sympy.Rational (for order verification)."""
        return self._c_sym

    def step(self, rhs: RHSProtocol, state: RKState, dt: float) -> RKState:
        t, u = state.t, state.u
        k: list[np.ndarray] = []
        for i in range(self._s):
            u_stage = u + dt * sum(self._A[i, j] * k[j] for j in range(i))
            k.append(rhs(t + self._c[i] * dt, u_stage))
        k_arr = np.stack(k, axis=0)
        u_new = u + dt * (self._b @ k_arr)
        return RKState(t + dt, u_new)


# ---------------------------------------------------------------------------
# Named instances — standard Butcher tableaux with sympy.Rational coefficients
# ---------------------------------------------------------------------------

forward_euler = RungeKuttaIntegrator(
    A=[[0]],
    b=[1],
    c=[0],
    order=1,
)

midpoint = RungeKuttaIntegrator(
    A=[[0, 0], ["1/2", 0]],
    b=[0, 1],
    c=[0, "1/2"],
    order=2,
)

heun = RungeKuttaIntegrator(
    A=[[0, 0], [1, 0]],
    b=["1/2", "1/2"],
    c=[0, 1],
    order=2,
)

ralston = RungeKuttaIntegrator(
    A=[[0, 0], ["2/3", 0]],
    b=["1/4", "3/4"],
    c=[0, "2/3"],
    order=2,
)

rk4 = RungeKuttaIntegrator(
    A=[
        [0, 0, 0, 0],
        ["1/2", 0, 0, 0],
        [0, "1/2", 0, 0],
        [0, 0, 1, 0],
    ],
    b=["1/6", "1/3", "1/3", "1/6"],
    c=[0, "1/2", "1/2", 1],
    order=4,
)

# Dormand-Prince (DOPRI5) — 6-stage 5th-order method with embedded 4th-order
# estimate.  Only the 5th-order weights are used for stepping here; the
# embedded pair becomes active when PIController is added in Phase 1.
dormand_prince = RungeKuttaIntegrator(
    A=[
        [0, 0, 0, 0, 0, 0],
        ["1/5", 0, 0, 0, 0, 0],
        ["3/40", "9/40", 0, 0, 0, 0],
        ["44/45", "-56/15", "32/9", 0, 0, 0],
        ["19372/6561", "-25360/2187", "64448/6561", "-212/729", 0, 0],
        ["9017/3168", "-355/33", "46732/5247", "49/176", "-5103/18656", 0],
    ],
    b=["35/384", 0, "500/1113", "125/192", "-2187/6784", "11/84"],
    c=[0, "1/5", "3/10", "4/5", "8/9", 1],
    order=5,
)

# Bogacki-Shampine (BS23) — 3-stage 3rd-order method with embedded 2nd-order
# estimate.  FSAL property: k₄ = k₁ of the next step (not exploited until Phase 1).
bogacki_shampine = RungeKuttaIntegrator(
    A=[
        [0, 0, 0],
        ["1/2", 0, 0],
        [0, "3/4", 0],
    ],
    b=["2/9", "1/3", "4/9"],
    c=[0, "1/2", "3/4"],
    order=3,
)


__all__ = [
    "RungeKuttaIntegrator",
    "bogacki_shampine",
    "dormand_prince",
    "forward_euler",
    "heun",
    "midpoint",
    "ralston",
    "rk4",
]
