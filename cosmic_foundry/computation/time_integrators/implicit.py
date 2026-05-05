"""Implicit Runge-Kutta integrators.

Orders 1-3 use diagonally implicit RK (DIRK / SDIRK) tableaux, where each stage
kᵢ solves the implicit equation

    kᵢ = f(tₙ + cᵢh, uₙ + h·Σⱼ≤ᵢ Aᵢⱼkⱼ)

via Newton iteration, requiring the Jacobian J = ∂f/∂u.  The Newton linear
system at each stage is

    (I − γᵢh·J(tᵢ, yᵢ))·δ = −(yᵢ − γᵢh·f(tᵢ, yᵢ) − y_exp)

where γᵢ = Aᵢᵢ and y_exp = uₙ + h·Σⱼ<ᵢ Aᵢⱼkⱼ is the explicit part.

Orders 4-6 use coupled collocation implicit RK methods.  The unknowns are the
stage values Yᵢ themselves and Newton iteration solves the full block system

    Yᵢ − uₙ − h·Σⱼ Aᵢⱼ f(tₙ + cⱼh, Yⱼ) = 0.

B-series order conditions are identical to explicit RK — the same rooted-tree
framework from bseries.py applies unchanged.

The canonical tableau for each order is:
    1 — backward Euler (L-stable)
    2 — implicit midpoint (A-stable, energy-conserving; Gauss-Legendre)
    3 — Crouzeix 1979, 2-stage L-stable SDIRK, γ = (3 + √3) / 6
    4 — 2-stage Gauss-Legendre collocation
    5 — 3-stage Radau IIA collocation
    6 — 3-stage Gauss-Legendre collocation
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

import sympy

from cosmic_foundry.computation.solvers._root_execution import solve_root_relation
from cosmic_foundry.computation.solvers.newton_root_solver import (
    DirectionalDerivativeRootRelation,
    RootRelation,
)
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    RHSProtocol,
    TimeIntegrator,
)
from cosmic_foundry.computation.time_integrators.solve_relation import (
    DirectionalDerivativeRHSProtocol,
    JacobianRHSProtocol,
    dirk_stage_directional_derivative_root_relation,
    dirk_stage_root_relation,
    implicit_stage_directional_derivative_root_relation,
    implicit_stage_root_relation,
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


@runtime_checkable
class ConstrainedNewtonRHSProtocol(Protocol):
    """Protocol for RHS objects that expose active constraint gradients."""

    def constraint_gradients(
        self,
        active: frozenset[int],
        t: float,
        u: Tensor,
        eps: float = 1e-7,
    ) -> Tensor | None:
        """Return gradient rows for active algebraic constraints."""
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


def _build_dirk_tableaux() -> dict:
    # Crouzeix (1979) 2-stage order-3 L-stable DIRK.
    # γ = (3 + √3) / 6 satisfies the order-3 Butcher conditions exactly.
    g = sympy.Rational(3, 6) + sympy.sqrt(3) / 6
    return {
        1: dict(A=[[1]], b=[1], c=[1]),
        2: dict(A=[["1/2"]], b=[1], c=["1/2"]),
        3: dict(
            A=[[g, 0], [1 - 2 * g, g]],
            b=["1/2", "1/2"],
            c=[g, 1 - g],
        ),
    }


_DIRK_TABLEAUX = _build_dirk_tableaux()


def _build_collocation_tableaux() -> dict:
    sqrt3 = sympy.sqrt(3)
    sqrt6 = sympy.sqrt(6)
    sqrt15 = sympy.sqrt(15)
    return {
        4: dict(
            A=[
                [sympy.Rational(1, 4), sympy.Rational(1, 4) - sqrt3 / 6],
                [sympy.Rational(1, 4) + sqrt3 / 6, sympy.Rational(1, 4)],
            ],
            b=[sympy.Rational(1, 2), sympy.Rational(1, 2)],
            c=[sympy.Rational(1, 2) - sqrt3 / 6, sympy.Rational(1, 2) + sqrt3 / 6],
        ),
        5: dict(
            A=[
                [
                    (88 - 7 * sqrt6) / 360,
                    (296 - 169 * sqrt6) / 1800,
                    (-2 + 3 * sqrt6) / 225,
                ],
                [
                    (296 + 169 * sqrt6) / 1800,
                    (88 + 7 * sqrt6) / 360,
                    (-2 - 3 * sqrt6) / 225,
                ],
                [
                    (16 - sqrt6) / 36,
                    (16 + sqrt6) / 36,
                    sympy.Rational(1, 9),
                ],
            ],
            b=[(16 - sqrt6) / 36, (16 + sqrt6) / 36, sympy.Rational(1, 9)],
            c=[(4 - sqrt6) / 10, (4 + sqrt6) / 10, sympy.Integer(1)],
        ),
        6: dict(
            A=[
                [
                    sympy.Rational(5, 36),
                    sympy.Rational(2, 9) - sqrt15 / 15,
                    sympy.Rational(5, 36) - sqrt15 / 30,
                ],
                [
                    sympy.Rational(5, 36) + sqrt15 / 24,
                    sympy.Rational(2, 9),
                    sympy.Rational(5, 36) - sqrt15 / 24,
                ],
                [
                    sympy.Rational(5, 36) + sqrt15 / 30,
                    sympy.Rational(2, 9) + sqrt15 / 15,
                    sympy.Rational(5, 36),
                ],
            ],
            b=[sympy.Rational(5, 18), sympy.Rational(4, 9), sympy.Rational(5, 18)],
            c=[
                sympy.Rational(1, 2) - sqrt15 / 10,
                sympy.Rational(1, 2),
                sympy.Rational(1, 2) + sqrt15 / 10,
            ],
        ),
    }


_COLLOCATION_TABLEAUX = _build_collocation_tableaux()


class ImplicitRungeKuttaIntegrator(TimeIntegrator):
    """Implicit Runge-Kutta method selected by convergence order.

    Orders 1-3 use sequential DIRK stage solves.  Orders 4-6 use coupled
    collocation stage solves over the full block system.
    Coefficients are stored as sympy.Rational (or general sympy expressions
    for irrational entries) for B-series verification, and as Python floats
    for numerical evaluation.

    The canonical tableau for each order is:
        1 — backward Euler (L-stable)
        2 — implicit midpoint (A-stable, energy-conserving)
        3 — Crouzeix 1979, 2-stage L-stable SDIRK
        4 — 2-stage Gauss-Legendre collocation
        5 — 3-stage Radau IIA collocation
        6 — 3-stage Gauss-Legendre collocation

    Parameters
    ----------
    order:
        Convergence order.  Must be one of {1, 2, 3, 4, 5, 6}.
    """

    def __init__(self, order: int) -> None:
        tableaux = _DIRK_TABLEAUX | _COLLOCATION_TABLEAUX
        if order not in tableaux:
            raise ValueError(
                f"ImplicitRungeKuttaIntegrator order must be one of "
                f"{sorted(tableaux)}, got {order}"
            )
        tab = tableaux[order]
        A, b, c = tab["A"], tab["b"], tab["c"]

        self._A_sym = [[sympy.sympify(a) for a in row] for row in A]
        self._b_sym = [sympy.sympify(bi) for bi in b]
        self._c_sym = [sympy.sympify(ci) for ci in c]
        self._order = order

        self._A_f = [[float(a) for a in row] for row in self._A_sym]
        self._b_f = [float(bi) for bi in self._b_sym]
        self._c_f = [float(ci) for ci in self._c_sym]
        self._s = len(b)
        self._coupled = order in _COLLOCATION_TABLEAUX

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

        ``rhs`` must expose either a Jacobian or Jacobian-vector products.
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
        jacobian_rhs = isinstance(rhs, JacobianRHSProtocol)
        jvp_rhs = isinstance(rhs, DirectionalDerivativeRHSProtocol)
        if not jacobian_rhs and not jvp_rhs:
            raise TypeError(
                "ImplicitRungeKuttaIntegrator requires Jacobian or JVP evidence; "
                f"got {type(rhs)}"
            )
        if self._coupled:
            if constraint_gradients is not None:
                raise ValueError(
                    "constraint-projected Newton is only implemented for DIRK orders"
                )
            return self._step_coupled(
                cast(JacobianRHSProtocol | DirectionalDerivativeRHSProtocol, rhs),
                state,
                dt,
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

            gamma_dt = gamma_i * dt

            relation: RootRelation | DirectionalDerivativeRootRelation
            if jvp_rhs:
                relation = dirk_stage_directional_derivative_root_relation(
                    cast(DirectionalDerivativeRHSProtocol, rhs),
                    y_exp,
                    t_i,
                    gamma_dt,
                )
            else:
                relation = dirk_stage_root_relation(
                    cast(JacobianRHSProtocol, rhs),
                    y_exp,
                    t_i,
                    gamma_dt,
                    equality_constraint_gradients=constraint_gradients,
                )
            y = solve_root_relation(relation)
            k.append(rhs(t_i, y))

        u_new = u
        for i in range(self._s):
            u_new = u_new + (self._b_f[i] * dt) * k[i]
        return state._replace(t=t + dt, u=u_new, dt=dt, err=0.0, history=None)

    def _step_coupled(
        self,
        rhs: JacobianRHSProtocol | DirectionalDerivativeRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        """Advance one fully implicit collocation step by block Newton."""
        t, u = state.t, state.u
        n = u.shape[0]
        s = self._s
        backend = u.backend
        relation: RootRelation | DirectionalDerivativeRootRelation
        if isinstance(rhs, DirectionalDerivativeRHSProtocol):
            relation = implicit_stage_directional_derivative_root_relation(
                self,
                rhs,
                state,
                dt,
            )
        else:
            relation = implicit_stage_root_relation(self, rhs, state, dt)
        stages = _unflatten_blocks(
            solve_root_relation(relation),
            s,
            n,
            backend=backend,
        )
        stage_times = [t + c_i * dt for c_i in self._c_f]
        f_vals = [rhs(stage_times[i], stages[i]) for i in range(s)]
        u_new = u
        for i in range(s):
            u_new = u_new + (self._b_f[i] * dt) * f_vals[i]
        return state._replace(t=t + dt, u=u_new, dt=dt, err=0.0, history=None)


def _unflatten_blocks(
    vec: Tensor,
    n_blocks: int,
    block_size: int,
    *,
    backend: Any,
) -> list[Tensor]:
    return [
        Tensor(
            [float(vec[b * block_size + i]) for i in range(block_size)],
            backend=backend,
        )
        for b in range(n_blocks)
    ]


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
    "FiniteDiffJacobianRHS",
    "JacobianRHS",
    "ImplicitRungeKuttaIntegrator",
    "ConstrainedNewtonRHSProtocol",
    "WithJacobianRHSProtocol",
    "stability_function",
]
