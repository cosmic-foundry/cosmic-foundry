"""Explicit Runge-Kutta integrators parameterized by Butcher tableaux."""

from __future__ import annotations

import sympy

from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    RHSProtocol,
    TimeIntegrator,
)


def _build_rk_tableaux() -> dict:
    return {
        1: dict(A=[[0]], b=[1], c=[0]),
        2: dict(A=[[0, 0], ["1/2", 0]], b=[0, 1], c=[0, "1/2"]),
        # Bogacki-Shampine BS23(2) — 4-stage FSAL; embedded pair at order 2.
        3: dict(
            A=[
                [0, 0, 0, 0],
                ["1/2", 0, 0, 0],
                [0, "3/4", 0, 0],
                ["2/9", "1/3", "4/9", 0],
            ],
            b=["2/9", "1/3", "4/9", 0],
            b_hat=["7/24", "1/4", "1/3", "1/8"],
            c=[0, "1/2", "3/4", 1],
        ),
        4: dict(
            A=[[0, 0, 0, 0], ["1/2", 0, 0, 0], [0, "1/2", 0, 0], [0, 0, 1, 0]],
            b=["1/6", "1/3", "1/3", "1/6"],
            c=[0, "1/2", "1/2", 1],
        ),
        # Dormand-Prince DOPRI5(4) — 7-stage FSAL; embedded pair at order 4.
        5: dict(
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
        ),
        # Butcher's 7-stage sixth-order explicit RK method.
        6: dict(
            A=[
                [0, 0, 0, 0, 0, 0, 0],
                ["1/3", 0, 0, 0, 0, 0, 0],
                [0, "2/3", 0, 0, 0, 0, 0],
                ["1/12", "1/3", "-1/12", 0, 0, 0, 0],
                ["-1/16", "9/8", "-3/16", "-3/8", 0, 0, 0],
                [0, "9/8", "-3/8", "-3/4", "1/2", 0, 0],
                ["9/44", "-9/11", "63/44", "18/11", 0, "-16/11", 0],
            ],
            b=["11/120", 0, "27/40", "27/40", "-4/15", "-4/15", "11/120"],
            c=[0, "1/3", "2/3", "1/3", "1/2", "1/2", 1],
        ),
    }


_RK_TABLEAUX = _build_rk_tableaux()


class RungeKuttaIntegrator(TimeIntegrator):
    """Explicit Runge-Kutta method selected by convergence order.

    Given du/dt = f(t, u), advances u from tₙ to tₙ₊₁ = tₙ + h via:

        kᵢ = f(tₙ + cᵢ h, uₙ + h Σⱼ<ᵢ Aᵢⱼ kⱼ)   i = 1, …, s
        uₙ₊₁ = uₙ + h Σᵢ bᵢ kᵢ

    The canonical tableau for each order is:
        1 — forward Euler
        2 — explicit midpoint
        3 — Bogacki-Shampine BS23 (4-stage FSAL, embedded pair at order 2)
        4 — classical RK4
        5 — Dormand-Prince DOPRI5 (7-stage FSAL, embedded pair at order 4)
        6 — Butcher 7-stage sixth-order method

    For orders 3 and 5, the embedded lower-order solution is computed during
    each step at no extra cost; the L² error estimate is stored in ODEState.err
    and used by PIController for adaptive step-size control.

    Coefficients are stored as sympy.Rational (or exact sympy expressions for
    irrational entries) for the B-series order-check framework, and as Python
    floats for numerical stepping.

    Parameters
    ----------
    order:
        Convergence order.  Must be one of {1, 2, 3, 4, 5, 6}.
    """

    def __init__(self, order: int) -> None:
        if order not in _RK_TABLEAUX:
            raise ValueError(
                f"RungeKuttaIntegrator order must be one of "
                f"{sorted(_RK_TABLEAUX)}, got {order}"
            )
        tab = _RK_TABLEAUX[order]
        A, b, c = tab["A"], tab["b"], tab["c"]
        b_hat = tab.get("b_hat")

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


__all__ = ["RungeKuttaIntegrator"]
