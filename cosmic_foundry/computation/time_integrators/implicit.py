"""Diagonally Implicit Runge-Kutta (DIRK / SDIRK) integrators.

DIRK methods share the same Butcher tableau structure as explicit RK but allow
non-zero diagonal entries in A.  Each stage kᵢ solves the implicit equation

    kᵢ = f(tₙ + cᵢh, uₙ + h·Σⱼ≤ᵢ Aᵢⱼkⱼ)

via Newton iteration, requiring the Jacobian J = ∂f/∂u.  The Newton linear
system at each stage is

    (I − γᵢh·J(tᵢ, yᵢ))·δ = −(yᵢ − γᵢh·f(tᵢ, yᵢ) − y_exp)

where γᵢ = Aᵢᵢ and y_exp = uₙ + h·Σⱼ<ᵢ Aᵢⱼkⱼ is the explicit part.

B-series order conditions are identical to explicit RK — the same rooted-tree
framework from bseries.py applies unchanged, since the algebraic order
conditions depend only on the tableau entries (A, b, c), not on the implicit
vs. explicit character of individual stages.

Named instances
---------------
backward_euler   — order 1, L-stable
implicit_midpoint — order 2, A-stable (Gauss-Legendre, energy-conserving)
crouzeix_3       — order 3, L-stable (Crouzeix 1979, γ = (3+√3)/6)
"""

from __future__ import annotations

import warnings
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
class WithJacobianRHSProtocol(Protocol):
    """Protocol for ODE right-hand sides that also expose a Jacobian.

    Extends the plain RHSProtocol with a `jacobian` method returning the
    n×n matrix J(t, u) = ∂f/∂u as a Tensor.  DIRK stage solvers require
    this to form the Newton iteration matrix I − γh·J.
    """

    def __call__(self, t: float, u: Tensor) -> Tensor:
        """Evaluate f(t, u)."""
        ...

    def jacobian(self, t: float, u: Tensor) -> Tensor:
        """Return the n×n Jacobian ∂f/∂u at (t, u) as a Tensor."""
        ...


class JacobianRHS:
    """WithJacobianRHSProtocol wrapping explicit f and Jacobian callables.

    Parameters
    ----------
    f:
        Callable (t, u) → Tensor giving the ODE right-hand side.
    jac:
        Callable (t, u) → Tensor giving the n×n Jacobian ∂f/∂u.
    """

    def __init__(
        self,
        f: Callable[[float, Tensor], Tensor],
        jac: Callable[[float, Tensor], Tensor],
    ) -> None:
        self._f = f
        self._jac = jac

    def __call__(self, t: float, u: Tensor) -> Tensor:
        result: Tensor = self._f(t, u)
        return result

    def jacobian(self, t: float, u: Tensor) -> Tensor:
        result: Tensor = self._jac(t, u)
        return result


class FiniteDiffJacobianRHS:
    """WithJacobianRHSProtocol that computes the Jacobian via forward differences.

    Parameters
    ----------
    f:
        Callable (t, u) → Tensor giving the ODE right-hand side.
    eps:
        Finite-difference step size.  Default 1e-7 gives ~7 correct digits
        for double-precision f with bounded second derivatives.
    """

    def __init__(
        self,
        f: Callable[[float, Tensor], Tensor],
        eps: float = 1e-7,
    ) -> None:
        self._f = f
        self._eps = eps

    def __call__(self, t: float, u: Tensor) -> Tensor:
        result: Tensor = self._f(t, u)
        return result

    def jacobian(self, t: float, u: Tensor) -> Tensor:
        n = u.shape[0]
        backend = u.backend
        f0 = self._f(t, u)
        cols: list[Tensor] = []
        for j in range(n):
            e_j = Tensor.zeros(n, backend=backend)
            e_j = e_j.set(j, Tensor(self._eps, backend=backend))
            col = (self._f(t, u + e_j) - f0) * (1.0 / self._eps)
            cols.append(col)
        rows = [[float(cols[j][i]) for j in range(n)] for i in range(n)]
        return Tensor(rows, backend=backend)


class ImplicitRungeKuttaIntegrator(TimeIntegrator):
    """Diagonally Implicit Runge-Kutta method defined by a Butcher tableau.

    Each stage of a DIRK step is solved by Newton iteration using the
    supplied LU factorizer to resolve the linear system at each Newton step.
    Coefficients are stored as sympy.Rational (or general sympy expressions
    for irrational entries) for B-series verification, and as Python floats
    for numerical evaluation.

    Parameters
    ----------
    A:
        Stage interaction matrix, shape (s, s), lower-triangular (diagonal
        entries may be non-zero).
    b:
        Quadrature weights, shape (s,).
    c:
        Abscissae, shape (s,).
    order:
        Declared convergence order.
    """

    def __init__(
        self,
        A: list[list],
        b: list,
        c: list,
        order: int,
    ) -> None:
        self._A_sym = [[sympy.sympify(a) for a in row] for row in A]
        self._b_sym = [sympy.sympify(bi) for bi in b]
        self._c_sym = [sympy.sympify(ci) for ci in c]
        self._order = order

        self._A_f = [[float(a) for a in row] for row in self._A_sym]
        self._b_f = [float(bi) for bi in self._b_sym]
        self._c_f = [float(ci) for ci in self._c_sym]
        self._s = len(b)

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._order

    @property
    def A_sym(self) -> list[list[sympy.Expr]]:
        """Butcher A matrix as sympy expressions (for B-series verification)."""
        return self._A_sym

    @property
    def b_sym(self) -> list[sympy.Expr]:
        """Quadrature weights as sympy expressions."""
        return self._b_sym

    @property
    def c_sym(self) -> list[sympy.Expr]:
        """Abscissae as sympy expressions."""
        return self._c_sym

    def step(
        self,
        rhs: RHSProtocol,
        state: ODEState,
        dt: float,
        *,
        constraint_gradients: Tensor | None = None,
    ) -> ODEState:
        """Advance state by one step of size dt via DIRK Newton iteration.

        ``rhs`` must satisfy ``WithJacobianRHSProtocol`` (expose ``.jacobian``).
        Returns a new ODEState with t = state.t + dt and updated u.
        The err field is set to 0.0 (no embedded pair in this base class).

        Parameters
        ----------
        constraint_gradients:
            Optional k × n matrix whose rows are the gradients of the k
            active algebraic constraints.  When provided, each Newton step
            is projected onto the null space of these gradients before being
            applied.  The ``active_constraints`` field of ``state`` is
            preserved in the returned state unchanged.
        """
        if not isinstance(rhs, WithJacobianRHSProtocol):
            raise TypeError(
                "ImplicitRungeKuttaIntegrator requires a WithJacobianRHSProtocol; "
                f"got {type(rhs)}"
            )
        t, u = state.t, state.u
        k: list[Tensor] = []

        for i in range(self._s):
            gamma_i = self._A_f[i][i]
            t_i = t + self._c_f[i] * dt
            # Explicit contribution from previously solved stages.
            y_exp = u
            for j in range(i):
                y_exp = y_exp + (self._A_f[i][j] * dt) * k[j]

            y = newton_solve(
                y_exp,
                gamma_i * dt,
                f=lambda y, _t=t_i: rhs(_t, y),  # type: ignore[misc]
                jac=lambda y, _t=t_i: rhs.jacobian(_t, y),  # type: ignore[misc]
                constraint_gradients=constraint_gradients,
            )
            k.append(rhs(t_i, y))

        u_new = u
        for i in range(self._s):
            u_new = u_new + (self._b_f[i] * dt) * k[i]
        return state._replace(t=t + dt, u=u_new, dt=dt, err=0.0, history=None)


class DIRKIntegrator(ImplicitRungeKuttaIntegrator):
    """Deprecated alias for ``ImplicitRungeKuttaIntegrator``."""

    def __init__(
        self,
        A: list[list],
        b: list,
        c: list,
        order: int,
    ) -> None:
        warnings.warn(
            "DIRKIntegrator is deprecated; use ImplicitRungeKuttaIntegrator.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(A, b, c, order)


# ---------------------------------------------------------------------------
# Named instances
# ---------------------------------------------------------------------------

backward_euler = ImplicitRungeKuttaIntegrator(
    A=[[1]],
    b=[1],
    c=[1],
    order=1,
)

implicit_midpoint = ImplicitRungeKuttaIntegrator(
    A=[["1/2"]],
    b=[1],
    c=["1/2"],
    order=2,
)

# Crouzeix (1979) 2-stage order-3 L-stable DIRK.
# γ = (3 + √3) / 6 satisfies the order-3 Butcher conditions exactly.
_gamma_c3 = sympy.Rational(3, 6) + sympy.sqrt(3) / 6
crouzeix_3 = ImplicitRungeKuttaIntegrator(
    A=[[_gamma_c3, 0], [1 - 2 * _gamma_c3, _gamma_c3]],
    b=["1/2", "1/2"],
    c=[_gamma_c3, 1 - _gamma_c3],
    order=3,
)


# ---------------------------------------------------------------------------
# Stability-function utilities
# ---------------------------------------------------------------------------


def stability_function(
    A_sym: list[list[sympy.Expr]],
    b_sym: list[sympy.Expr],
) -> sympy.Expr:
    """Return R(z) = 1 + z·bᵀ·(I − zA)⁻¹·e as a sympy rational function.

    R(z) is the stability function of the RK method: a step of size h on
    the Dahlquist test equation dy/dt = λy gives y_{n+1} = R(λh)·y_n.
    A-stability requires |R(z)| ≤ 1 for all Re(z) ≤ 0.
    L-stability additionally requires R(z) → 0 as z → −∞.
    """
    z = sympy.Symbol("z")
    s = len(b_sym)
    A_mat = sympy.Matrix(A_sym)
    b_vec = sympy.Matrix(b_sym)
    e_vec = sympy.ones(s, 1)
    I_mat = sympy.eye(s)
    inv = (I_mat - z * A_mat).inv()
    R = 1 + z * (b_vec.T * inv * e_vec)[0, 0]
    return sympy.simplify(R)


__all__ = [
    "DIRKIntegrator",
    "FiniteDiffJacobianRHS",
    "JacobianRHS",
    "ImplicitRungeKuttaIntegrator",
    "WithJacobianRHSProtocol",
    "backward_euler",
    "crouzeix_3",
    "implicit_midpoint",
    "stability_function",
]
