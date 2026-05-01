"""Exponential time integrators for semilinear ODE splits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators.integrator import ODEState
from cosmic_foundry.computation.time_integrators.runge_kutta import RungeKuttaIntegrator

_SERIES_TOL = 1e-14
_SERIES_MAX_TERMS = 80


@runtime_checkable
class SemilinearRHSProtocol(Protocol):
    """Protocol for semilinear ODEs ``du/dt = L u + N(t, u)``."""

    @property
    def linear_operator(self) -> Tensor:
        """Dense matrix ``L`` for the exactly treated linear part."""
        ...

    def nonlinear(self, t: float, u: Tensor) -> Tensor:
        """Evaluate the explicit nonlinear or residual component ``N(t, u)``."""
        ...


class SemilinearRHS:
    """Concrete ``SemilinearRHSProtocol`` from a matrix and callable."""

    def __init__(
        self,
        linear_operator: Tensor,
        nonlinear: Callable[[float, Tensor], Tensor],
    ) -> None:
        self._linear_operator = linear_operator
        self._nonlinear = nonlinear

    @property
    def linear_operator(self) -> Tensor:
        return self._linear_operator

    def nonlinear(self, t: float, u: Tensor) -> Tensor:
        result: Tensor = self._nonlinear(t, u)
        return result


class PhiFunction:
    """Operator-valued ``phi_k`` coefficient action.

    ``phi_k(A) v = Σ_{m≥0} A^m v / (m+k)!``.

    The implementation applies the series directly to vectors.  This is the
    small dense path used by Phase 13 tests; Krylov projection for large
    matrices is a later performance specialization, not a different API.
    """

    def __init__(self, k: int) -> None:
        if k < 0:
            raise ValueError("phi-function index k must be non-negative.")
        self.k = k

    def apply(self, A: Tensor, v: Tensor) -> Tensor:
        """Return ``phi_k(A) v``."""
        if self.k == 0:
            return _matrix_exp_action(A, v)

        term = v * (1.0 / _factorial(self.k))
        result = term
        for m in range(1, _SERIES_MAX_TERMS):
            term = (A @ term) * (1.0 / (m + self.k))
            result = result + term
            if float(norm(term)) <= _SERIES_TOL * (1.0 + float(norm(result))):
                return result
        return result


class LawsonRungeKuttaIntegrator:
    """Integrating-factor RK method for semilinear systems.

    For ``u' = L u + N(t, u)`` with constant dense ``L``, this applies the
    change of variables ``v(t) = exp(-tL) u(t)`` and advances ``v`` with the
    explicit RK tableau of the requested order.  Written in ``u`` variables,
    each stage is transported by exact linear-flow actions ``exp(a h L)``.

    The method has the same order as the underlying RK tableau when the
    matrix-exponential actions are exact.  This implementation uses the same
    dense Taylor action as ``PhiFunction(0)``, which is the small-system path
    used throughout this module.

    Parameters
    ----------
    order:
        Convergence order.  Must be one of {1, 2, 3, 4, 5, 6}.
    """

    def __init__(self, order: int) -> None:
        rk = RungeKuttaIntegrator(order)
        self._A_f = [[float(a) for a in row] for row in rk.A_sym]
        self._b_f = [float(b) for b in rk.b_sym]
        self._c_f = [float(c) for c in rk.c_sym]
        self._order = order
        self._s = len(self._b_f)

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._order

    def step(
        self,
        rhs: SemilinearRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        """Advance one Lawson RK step for ``u' = L u + N(t, u)``."""
        L = rhs.linear_operator
        t, u = state.t, state.u
        exp_action = PhiFunction(0)
        nonlinear_stages: list[Tensor] = []

        for i in range(self._s):
            y = exp_action.apply((self._c_f[i] * dt) * L, u)
            for j in range(i):
                a_ij = self._A_f[i][j]
                if a_ij != 0.0:
                    y = y + (a_ij * dt) * exp_action.apply(
                        ((self._c_f[i] - self._c_f[j]) * dt) * L,
                        nonlinear_stages[j],
                    )
            nonlinear_stages.append(rhs.nonlinear(t + self._c_f[i] * dt, y))

        u_new = exp_action.apply(dt * L, u)
        for i in range(self._s):
            if self._b_f[i] != 0.0:
                u_new = u_new + (self._b_f[i] * dt) * exp_action.apply(
                    ((1.0 - self._c_f[i]) * dt) * L,
                    nonlinear_stages[i],
                )
        return ODEState(t + dt, u_new, dt, 0.0)


def _matrix_exp_action(A: Tensor, v: Tensor) -> Tensor:
    """Return ``exp(A) v`` by a small dense Taylor action."""
    term = v
    result = term
    for m in range(1, _SERIES_MAX_TERMS):
        term = (A @ term) * (1.0 / m)
        result = result + term
        if float(norm(term)) <= _SERIES_TOL * (1.0 + float(norm(result))):
            return result
    return result


def _factorial(n: int) -> int:
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


__all__ = [
    "LawsonRungeKuttaIntegrator",
    "PhiFunction",
    "SemilinearRHS",
    "SemilinearRHSProtocol",
]
