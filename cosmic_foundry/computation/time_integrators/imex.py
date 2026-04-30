"""IMEX additive Runge-Kutta integrators for split f_E + f_I problems.

An IMEX-RK method decomposes the ODE RHS as f(t, u) = f_E(t, u) + f_I(t, u),
applying an explicit scheme to the non-stiff component f_E and an implicit
DIRK scheme to the stiff component f_I.  Each stage i couples both parts
at the same stage value yᵢ:

    yᵢ − γᵢ·h·f_I(tᵢ, yᵢ) = y_exp,i

where γᵢ = Aᴵ[i,i] and y_exp,i accumulates the explicit contributions from
Ã and the lower-triangular implicit contributions from Aᴵ.  Newton iteration
via LUFactorization resolves each stage using the Jacobian of f_I.

The canonical tableau for each order is:
    2 — ARS(2,2,2): Ascher-Ruuth-Spiteri 1997, A-stable implicit pair,
        γ = (2−√2)/2; implicit tableau is SDIRK; b_E = b_I.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators._newton import newton_solve
from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    RHSProtocol,
    TimeIntegrator,
)


@runtime_checkable
class SplitRHSProtocol(Protocol):
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


class SplitRHS:
    """Concrete SplitRHSProtocol wrapping three callables.

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


def _build_imex_tableaux() -> dict:
    # ARS(2,2,2): Ascher, Ruuth, Spiteri (1997), Section 3.
    # γ = (2−√2)/2 satisfies the order-2 Butcher conditions for the coupled pair.
    # Implicit tableau: SDIRK with constant diagonal γ.
    # δ = 1 − 1/(2γ) = −√2/2
    g = (sympy.Integer(2) - sympy.sqrt(2)) / 2
    d = 1 - 1 / (2 * g)
    return {
        2: dict(
            A_E=[[0, 0, 0], [g, 0, 0], [d, 1 - d, 0]],
            b_E=[0, 1 - g, g],
            c_E=[0, g, 1],
            A_I=[[0, 0, 0], [0, g, 0], [0, 1 - g, g]],
            b_I=[0, 1 - g, g],
            c_I=[0, g, 1],
        ),
    }


_IMEX_TABLEAUX = _build_imex_tableaux()


class AdditiveRungeKuttaIntegrator(TimeIntegrator):
    """IMEX additive Runge-Kutta method selected by convergence order.

    Uses an explicit tableau (A_E, b_E, c_E) for the non-stiff component
    and a DIRK tableau (A_I, b_I, c_I) for the stiff component.  Each stage
    yᵢ is found by Newton iteration on the implicit equation; the explicit
    component is then evaluated at the converged stage value.  When the
    implicit diagonal γᵢ = 0, the stage is purely explicit (no Newton step).

    The canonical tableau for each order is:
        2 — ARS(2,2,2): Ascher-Ruuth-Spiteri 1997, A-stable

    Parameters
    ----------
    order:
        Convergence order.  Must be one of {2}.
    """

    def __init__(self, order: int) -> None:
        if order not in _IMEX_TABLEAUX:
            raise ValueError(
                f"AdditiveRungeKuttaIntegrator order must be one of "
                f"{sorted(_IMEX_TABLEAUX)}, got {order}"
            )
        tab = _IMEX_TABLEAUX[order]
        A_E, b_E = tab["A_E"], tab["b_E"]
        A_I, b_I, c_I = tab["A_I"], tab["b_I"], tab["c_I"]

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
        rhs: RHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        """Advance state by one IMEX step of size dt.

        ``rhs`` must satisfy ``SplitRHSProtocol`` (expose ``.explicit``,
        ``.implicit``, ``.jacobian_implicit``).
        Returns a new ODEState with t = state.t + dt and updated u.
        The err field is 0.0 (no embedded pair in this base class).
        """
        if not isinstance(rhs, SplitRHSProtocol):
            raise TypeError(
                "AdditiveRungeKuttaIntegrator requires a SplitRHSProtocol; "
                f"got {type(rhs)}"
            )
        t, u = state.t, state.u
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
                y = newton_solve(
                    y_exp,
                    gamma_i * dt,
                    f=lambda y, _t=t_i: rhs.implicit(_t, y),  # type: ignore[misc]
                    jac=lambda y, _t=t_i: rhs.jacobian_implicit(_t, y),  # type: ignore[misc]
                )

            k_E.append(rhs.explicit(t_i, y))
            k_I.append(rhs.implicit(t_i, y))

        u_new = u
        for i in range(self._s):
            u_new = u_new + (self._b_E_f[i] * dt) * k_E[i]
            u_new = u_new + (self._b_I_f[i] * dt) * k_I[i]

        return ODEState(t + dt, u_new, dt, 0.0)


__all__ = [
    "AdditiveRungeKuttaIntegrator",
    "SplitRHS",
    "SplitRHSProtocol",
]
