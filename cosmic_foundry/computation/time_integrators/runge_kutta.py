"""Explicit Runge-Kutta integrators parameterized by Butcher tableaux."""

from __future__ import annotations

import sympy

from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    RHSProtocol,
    TimeIntegrator,
)


class RungeKuttaIntegrator(TimeIntegrator):
    """Explicit Runge-Kutta method defined by a Butcher tableau (A, b, c, order).

    Given du/dt = f(t, u), advances u from tₙ to tₙ₊₁ = tₙ + h via:

        kᵢ = f(tₙ + cᵢ h, uₙ + h Σⱼ<ᵢ Aᵢⱼ kⱼ)   i = 1, …, s
        uₙ₊₁ = uₙ + h Σᵢ bᵢ kᵢ

    When an embedded weight vector b_hat is provided, the embedded solution

        û_{n+1} = uₙ + h Σᵢ b̂ᵢ kᵢ

    is computed using the same stage evaluations at no extra cost.  The L2
    norm ‖uₙ₊₁ − û_{n+1}‖ is an O(hᵖ) local error estimate and is stored
    in the returned ODEState.err for use by PIController.

    Coefficients may be supplied as Python numbers or strings parseable by
    sympy.Rational (e.g. "1/2").  They are stored as sympy.Rational for the
    order-check framework and as Python floats for numerical stepping.

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
    b_hat:
        Embedded quadrature weights, shape (s,).  When provided, each step
        also computes the lower-order embedded solution and stores the error
        norm in ODEState.err.
    """

    def __init__(
        self,
        A: list[list],
        b: list,
        c: list,
        order: int,
        b_hat: list | None = None,
    ) -> None:
        self._A_sym = [[sympy.Rational(a) for a in row] for row in A]
        self._b_sym = [sympy.Rational(bi) for bi in b]
        self._c_sym = [sympy.Rational(ci) for ci in c]
        self._b_hat_sym = (
            [sympy.Rational(bi) for bi in b_hat] if b_hat is not None else None
        )
        self._order = order

        self._A_f = [[float(a) for a in row] for row in self._A_sym]
        self._b_f = [float(bi) for bi in self._b_sym]
        self._c_f = [float(ci) for ci in self._c_sym]
        self._b_hat_f = (
            [float(bi) for bi in self._b_hat_sym]
            if self._b_hat_sym is not None
            else None
        )
        self._s = len(b)

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

    @property
    def b_hat_sym(self) -> list[sympy.Rational] | None:
        """Embedded weights as sympy.Rational, or None if no embedded pair."""
        return self._b_hat_sym

    def step(self, rhs: RHSProtocol, state: ODEState, dt: float) -> ODEState:
        t, u = state.t, state.u
        k: list[Tensor] = []
        for i in range(self._s):
            u_stage = u
            for j in range(i):
                u_stage = u_stage + self._A_f[i][j] * k[j] * dt
            k.append(rhs(t + self._c_f[i] * dt, u_stage))
        u_new = u
        for i in range(self._s):
            u_new = u_new + self._b_f[i] * k[i] * dt
        err = 0.0
        if self._b_hat_f is not None:
            u_hat = u
            for i in range(self._s):
                u_hat = u_hat + self._b_hat_f[i] * k[i] * dt
            err = float(norm(u_new - u_hat))
        return ODEState(t + dt, u_new, dt, err)


# ---------------------------------------------------------------------------
# Named instances — standard Butcher tableaux with sympy.Rational coefficients
# ---------------------------------------------------------------------------

forward_euler = RungeKuttaIntegrator(
    A=[[0]],
    b=[1],
    c=[0],
    order=1,
)

# Explicit midpoint: b = [0, 1], c = [0, 1/2].
midpoint = RungeKuttaIntegrator(
    A=[[0, 0], ["1/2", 0]],
    b=[0, 1],
    c=[0, "1/2"],
    order=2,
)

# Heun (RK22) with Heun-Euler embedded pair: b̂ = forward Euler weights.
heun = RungeKuttaIntegrator(
    A=[[0, 0], [1, 0]],
    b=["1/2", "1/2"],
    b_hat=[1, 0],
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

# Dormand-Prince DOPRI5(4) — 7-stage explicit method with FSAL.
# Stage 7 is the FSAL stage: k₇ = f(t+h, u_{n+1}).  Its weight in b is 0
# so the primary 5th-order solution is unchanged from the 6-stage version.
# b_hat gives the 4th-order embedded solution using all 7 stages.
dormand_prince = RungeKuttaIntegrator(
    A=[
        [0, 0, 0, 0, 0, 0, 0],
        ["1/5", 0, 0, 0, 0, 0, 0],
        ["3/40", "9/40", 0, 0, 0, 0, 0],
        ["44/45", "-56/15", "32/9", 0, 0, 0, 0],
        ["19372/6561", "-25360/2187", "64448/6561", "-212/729", 0, 0, 0],
        ["9017/3168", "-355/33", "46732/5247", "49/176", "-5103/18656", 0, 0],
        ["35/384", 0, "500/1113", "125/192", "-2187/6784", "11/84", 0],
    ],
    b=["35/384", 0, "500/1113", "125/192", "-2187/6784", "11/84", 0],
    b_hat=[
        "5179/57600",
        0,
        "7571/16695",
        "393/640",
        "-92097/339200",
        "187/2100",
        "1/40",
    ],
    c=[0, "1/5", "3/10", "4/5", "8/9", 1, 1],
    order=5,
)

# Bogacki-Shampine BS23(2) — 4-stage method with FSAL.
# Stage 4 is the FSAL stage: k₄ = f(t+h, u_{n+1}).  Its weight in b is 0
# so the primary 3rd-order solution is unchanged from the 3-stage version.
# b_hat gives the 2nd-order embedded solution using all 4 stages.
bogacki_shampine = RungeKuttaIntegrator(
    A=[
        [0, 0, 0, 0],
        ["1/2", 0, 0, 0],
        [0, "3/4", 0, 0],
        ["2/9", "1/3", "4/9", 0],
    ],
    b=["2/9", "1/3", "4/9", 0],
    b_hat=["7/24", "1/4", "1/3", "1/8"],
    c=[0, "1/2", "3/4", 1],
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
