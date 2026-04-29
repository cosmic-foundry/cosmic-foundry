"""IMEX additive Runge-Kutta integrators for split f_E + f_I problems.

An IMEX-RK method decomposes the ODE RHS as f(t, u) = f_E(t, u) + f_I(t, u),
applying an explicit scheme to the non-stiff component f_E and an implicit
DIRK scheme to the stiff component f_I.  Each stage i couples both parts
at the same stage value yᵢ:

    yᵢ − γᵢ·h·f_I(tᵢ, yᵢ) = y_exp,i

where γᵢ = Aᴵ[i,i] and y_exp,i accumulates the explicit contributions from
Ã and the lower-triangular implicit contributions from Aᴵ.  Newton iteration
via LUFactorization resolves each stage using the Jacobian of f_I.

Named instances
---------------
ars222  — order 2, A-stable implicit pair (Ascher-Ruuth-Spiteri 1997)
          γ = (2−√2)/2; implicit tableau is SDIRK; explicit and implicit
          update weights are identical (b_E = b_I).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import sympy

from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators.integrator import RKState

_LU = LUFactorization()
_NEWTON_MAX_ITER = 50
_NEWTON_TOL = 1e-12


@runtime_checkable
class AdditiveRHSProtocol(Protocol):
    """Protocol for ODE right-hand sides with an explicit/implicit additive split.

    Implementers supply three callables: the non-stiff explicit component,
    the stiff implicit component, and the Jacobian of the implicit component.
    The full RHS is f = explicit + implicit.
    """

    def explicit(self, t: float, u: Tensor) -> Tensor:
        """Evaluate the non-stiff component f_E(t, u)."""
        ...

    def implicit(self, t: float, u: Tensor) -> Tensor:
        """Evaluate the stiff component f_I(t, u)."""
        ...

    def jacobian_implicit(self, t: float, u: Tensor) -> Tensor:
        """Return the n×n Jacobian ∂f_I/∂u at (t, u) as a Tensor."""
        ...


class AdditiveRHS:
    """Concrete AdditiveRHSProtocol wrapping three callables.

    Parameters
    ----------
    f_E:
        Callable (t, u) → Tensor giving the non-stiff explicit part.
    f_I:
        Callable (t, u) → Tensor giving the stiff implicit part.
    jac_I:
        Callable (t, u) → Tensor giving the n×n Jacobian ∂f_I/∂u.
    """

    def __init__(
        self,
        f_E: Callable[[float, Tensor], Tensor],
        f_I: Callable[[float, Tensor], Tensor],
        jac_I: Callable[[float, Tensor], Tensor],
    ) -> None:
        self._f_E = f_E
        self._f_I = f_I
        self._jac_I = jac_I

    def explicit(self, t: float, u: Tensor) -> Tensor:
        result: Tensor = self._f_E(t, u)
        return result

    def implicit(self, t: float, u: Tensor) -> Tensor:
        result: Tensor = self._f_I(t, u)
        return result

    def jacobian_implicit(self, t: float, u: Tensor) -> Tensor:
        result: Tensor = self._jac_I(t, u)
        return result


class IMEXIntegrator:
    """IMEX additive Runge-Kutta method defined by two Butcher tableaux.

    Uses an explicit tableau (A_E, b_E, c_E) for the non-stiff component
    and a DIRK tableau (A_I, b_I, c_I) for the stiff component.  Each stage
    yᵢ is found by Newton iteration on the implicit equation; the explicit
    component is then evaluated at the converged stage value.  When the
    implicit diagonal γᵢ = 0, the stage is purely explicit (no Newton step).

    Parameters
    ----------
    A_E:
        Explicit stage interaction matrix, shape (s, s), strictly lower-triangular.
    b_E:
        Explicit quadrature weights, shape (s,).
    c_E:
        Explicit abscissae, shape (s,).
    A_I:
        Implicit stage interaction matrix, shape (s, s), lower-triangular.
    b_I:
        Implicit quadrature weights, shape (s,).
    c_I:
        Implicit abscissae, shape (s,).
    order:
        Declared convergence order.
    """

    def __init__(
        self,
        A_E: list[list],
        b_E: list,
        c_E: list,
        A_I: list[list],
        b_I: list,
        c_I: list,
        order: int,
    ) -> None:
        self._A_E_f = [[float(sympy.sympify(a)) for a in row] for row in A_E]
        self._b_E_f = [float(sympy.sympify(bi)) for bi in b_E]
        self._A_I_f = [[float(sympy.sympify(a)) for a in row] for row in A_I]
        self._b_I_f = [float(sympy.sympify(bi)) for bi in b_I]
        self._c_I_f = [float(sympy.sympify(ci)) for ci in c_I]
        self._order = order
        self._s = len(b_E)

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._order

    def step(
        self,
        rhs: AdditiveRHSProtocol,
        state: RKState,
        dt: float,
    ) -> RKState:
        """Advance state by one IMEX step of size dt.

        Returns a new RKState with t = state.t + dt and updated u.
        The err field is 0.0 (no embedded pair in this base class).
        """
        t, u = state.t, state.u
        n = u.shape[0]
        backend = u.backend
        k_E: list[Tensor] = []
        k_I: list[Tensor] = []

        for i in range(self._s):
            gamma_i = self._A_I_f[i][i]
            t_i = t + self._c_I_f[i] * dt

            # Accumulate explicit-tableau and lower-triangular implicit contributions.
            y_exp = u
            for j in range(i):
                y_exp = y_exp + (self._A_E_f[i][j] * dt) * k_E[j]
                y_exp = y_exp + (self._A_I_f[i][j] * dt) * k_I[j]

            # Implicit solve for yᵢ; skip Newton when the diagonal is zero.
            if abs(gamma_i) < 1e-14:
                y = y_exp
            else:
                y = y_exp
                for _ in range(_NEWTON_MAX_ITER):
                    f_I_val = rhs.implicit(t_i, y)
                    r = y - (gamma_i * dt) * f_I_val - y_exp
                    if float(norm(r)) < _NEWTON_TOL * (1.0 + float(norm(y))):
                        break
                    J = rhs.jacobian_implicit(t_i, y)
                    M = Tensor.eye(n, backend=backend) - (gamma_i * dt) * J
                    delta = _LU.factorize(M).solve(Tensor.zeros(n, backend=backend) - r)
                    y = y + delta
                    if float(norm(delta)) < _NEWTON_TOL * (1.0 + float(norm(y))):
                        break

            k_E.append(rhs.explicit(t_i, y))
            k_I.append(rhs.implicit(t_i, y))

        u_new = u
        for i in range(self._s):
            u_new = u_new + (self._b_E_f[i] * dt) * k_E[i]
            u_new = u_new + (self._b_I_f[i] * dt) * k_I[i]

        return RKState(t + dt, u_new, dt, 0.0)


# ---------------------------------------------------------------------------
# Named instances
# ---------------------------------------------------------------------------

# ARS(2,2,2): Ascher, Ruuth, Spiteri (1997), Section 3.
# γ = (2−√2)/2 satisfies the order-2 Butcher conditions for the coupled pair.
# Implicit tableau: SDIRK with constant diagonal γ.
# Explicit tableau: second stage at t_n + γh with weight γ; third at t_n + h.
# Update weights b_E = b_I = [0, 1−γ, γ].
_g = (sympy.Integer(2) - sympy.sqrt(2)) / 2
_d = 1 - 1 / (2 * _g)  # δ = 1 − 1/(2γ) = −√2/2

ars222 = IMEXIntegrator(
    A_E=[[0, 0, 0], [_g, 0, 0], [_d, 1 - _d, 0]],
    b_E=[0, 1 - _g, _g],
    c_E=[0, _g, 1],
    A_I=[[0, 0, 0], [0, _g, 0], [0, 1 - _g, _g]],
    b_I=[0, 1 - _g, _g],
    c_I=[0, _g, 1],
    order=2,
)


__all__ = [
    "AdditiveRHS",
    "AdditiveRHSProtocol",
    "IMEXIntegrator",
    "ars222",
]
