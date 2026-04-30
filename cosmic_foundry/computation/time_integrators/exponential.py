"""Exponential time integrators for semilinear ODE splits."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators.integrator import ODEState

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


class ExponentialEulerIntegrator:
    """First-order ETD Euler integrator for ``u' = L u + N(t, u)``."""

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return 1

    def step(
        self,
        rhs: SemilinearRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        A = rhs.linear_operator * dt
        u = state.u
        Nu = rhs.nonlinear(state.t, u)
        u_new = PhiFunction(0).apply(A, u) + dt * PhiFunction(1).apply(A, Nu)
        return ODEState(state.t + dt, u_new, dt, 0.0)


class ETDRK2Integrator:
    """Second-order exponential Runge-Kutta integrator."""

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return 2

    def step(
        self,
        rhs: SemilinearRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        A = rhs.linear_operator * dt
        t, u = state.t, state.u
        phi0 = PhiFunction(0)
        phi1 = PhiFunction(1)
        phi2 = PhiFunction(2)

        Nu = rhs.nonlinear(t, u)
        e_u = phi0.apply(A, u)
        predictor = e_u + dt * phi1.apply(A, Nu)
        N_pred = rhs.nonlinear(t + dt, predictor)
        u_new = e_u + dt * (phi1.apply(A, Nu) + phi2.apply(A, N_pred - Nu))
        return ODEState(t + dt, u_new, dt, 0.0)


class CoxMatthewsETDRK4Integrator:
    """Fourth-order Cox-Matthews exponential Runge-Kutta integrator."""

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return 4

    def step(
        self,
        rhs: SemilinearRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        A = rhs.linear_operator * dt
        A_half = A * 0.5
        t, u = state.t, state.u

        phi0 = PhiFunction(0)
        phi1 = PhiFunction(1)

        Nu = rhs.nonlinear(t, u)
        e_half_u = phi0.apply(A_half, u)
        a = e_half_u + (0.5 * dt) * phi1.apply(A_half, Nu)
        Na = rhs.nonlinear(t + 0.5 * dt, a)

        b = e_half_u + (0.5 * dt) * phi1.apply(A_half, Na)
        Nb = rhs.nonlinear(t + 0.5 * dt, b)

        c = phi0.apply(A_half, a) + (0.5 * dt) * phi1.apply(A_half, 2.0 * Nb - Nu)
        Nc = rhs.nonlinear(t + dt, c)

        b1 = _phi_combination(A, Nu, c1=1.0, c2=-3.0, c3=4.0)
        b2 = _phi_combination(A, Na + Nb, c2=2.0, c3=-4.0)
        b4 = _phi_combination(A, Nc, c2=-1.0, c3=4.0)
        u_new = phi0.apply(A, u) + dt * (b1 + b2 + b4)
        return ODEState(t + dt, u_new, dt, 0.0)


def _phi_combination(
    A: Tensor,
    v: Tensor,
    *,
    c1: float = 0.0,
    c2: float = 0.0,
    c3: float = 0.0,
) -> Tensor:
    """Return ``(c1 phi1(A) + c2 phi2(A) + c3 phi3(A)) v``."""
    zero = Tensor.zeros(*v.shape, backend=v.backend)
    result = zero
    if c1 != 0.0:
        result = result + c1 * PhiFunction(1).apply(A, v)
    if c2 != 0.0:
        result = result + c2 * PhiFunction(2).apply(A, v)
    if c3 != 0.0:
        result = result + c3 * PhiFunction(3).apply(A, v)
    return result


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


class KrogstadETDRK4Integrator:
    """Fourth-order Krogstad exponential Runge-Kutta integrator.

    Same nodes and update weights as Cox-Matthews ETDRK4, but the internal
    stages U3 and U4 include φ2 corrections instead of φ1.  These corrections
    satisfy the Hochbruck-Ostermann stiff-order conditions that Cox-Matthews
    fails, giving full 4th-order accuracy on semilinear parabolic problems with
    stiff linear part.

    Reference: S. Krogstad, "Generalized integrating factor methods for stiff
    PDEs," J. Comput. Phys. 203(1):72-88, 2005.
    """

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return 4

    def step(
        self,
        rhs: SemilinearRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        A = rhs.linear_operator * dt
        A_half = A * 0.5
        t, u = state.t, state.u

        phi0 = PhiFunction(0)
        phi1 = PhiFunction(1)
        phi2 = PhiFunction(2)

        F1 = rhs.nonlinear(t, u)
        e_half_u = phi0.apply(A_half, u)
        e_u = phi0.apply(A, u)

        # Stage 2: identical to Cox-Matthews.
        U2 = e_half_u + (0.5 * dt) * phi1.apply(A_half, F1)
        F2 = rhs.nonlinear(t + 0.5 * dt, U2)

        # Stage 3: φ2 correction on the F2-F1 difference (absent in Cox-Matthews).
        # Cox-Matthews: U3 = φ0(A/2) u + (h/2) φ1(A/2) F2
        # Krogstad:     U3 = φ0(A/2) u + (h/2) φ1(A/2) F1 + h φ2(A/2) (F2 - F1)
        U3 = (
            e_half_u
            + (0.5 * dt) * phi1.apply(A_half, F1)
            + dt * phi2.apply(A_half, F2 - F1)
        )
        F3 = rhs.nonlinear(t + 0.5 * dt, U3)

        # Stage 4: φ2 correction at full step; built from u, not from U2.
        # Cox-Matthews: U4 = φ0(A/2) U2 + (h/2) φ1(A/2) (2 F3 - F1)
        # Krogstad:     U4 = φ0(A) u + h φ1(A) F1 + 2h φ2(A) (F3 - F1)
        U4 = e_u + dt * phi1.apply(A, F1) + 2.0 * dt * phi2.apply(A, F3 - F1)
        F4 = rhs.nonlinear(t + dt, U4)

        # Update weights are identical to Cox-Matthews.
        b1 = _phi_combination(A, F1, c1=1.0, c2=-3.0, c3=4.0)
        b2 = _phi_combination(A, F2 + F3, c2=2.0, c3=-4.0)
        b4 = _phi_combination(A, F4, c2=-1.0, c3=4.0)
        u_new = e_u + dt * (b1 + b2 + b4)
        return ODEState(t + dt, u_new, dt, 0.0)


etd_euler = ExponentialEulerIntegrator()
etdrk2 = ETDRK2Integrator()
cox_matthews_etdrk4 = CoxMatthewsETDRK4Integrator()
krogstad_etdrk4 = KrogstadETDRK4Integrator()


class LinearPlusNonlinearRHS(SemilinearRHS):
    """Deprecated alias for ``SemilinearRHS``."""

    def __init__(
        self,
        linear_operator: Tensor,
        nonlinear: Callable[[float, Tensor], Tensor],
    ) -> None:
        warnings.warn(
            "LinearPlusNonlinearRHS is deprecated; use SemilinearRHS.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(linear_operator, nonlinear)


__all__ = [
    "CoxMatthewsETDRK4Integrator",
    "ETDRK2Integrator",
    "ExponentialEulerIntegrator",
    "KrogstadETDRK4Integrator",
    "LinearPlusNonlinearRHS",
    "LinearPlusNonlinearRHSProtocol",
    "PhiFunction",
    "SemilinearRHS",
    "SemilinearRHSProtocol",
    "cox_matthews_etdrk4",
    "etd_euler",
    "etdrk2",
    "krogstad_etdrk4",
]

LinearPlusNonlinearRHSProtocol = SemilinearRHSProtocol
