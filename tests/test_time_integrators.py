"""Verification for the time-integration layer.

Three-tier auto-discovery framework for TimeIntegrator subclasses:

  _RKOrderClaim     — symbolic Butcher order conditions (Tier A)
  _ConvergenceClaim — temporal convergence rate on dy/dt = λy (Tier B)
  _StepperClaim     — end-to-end TimeStepper.advance accuracy (Tier C)

Each claim encodes both what is being verified and how to verify it.
Adding a new claim requires only appending to the relevant registry list;
the single parametric test per tier covers all entries.

Time-integrator verification is structurally distinct from spatial-operator
verification (different order conditions, different test problems, different
verification hierarchy), so this file is independent of test_convergence.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy

from cosmic_foundry.computation.time_integrators import (
    BlackBoxRHS,
    ConstantStep,
    RKState,
    RungeKuttaIntegrator,
    TimeStepper,
    bogacki_shampine,
    dormand_prince,
    forward_euler,
    heun,
    midpoint,
    ralston,
    rk4,
)
from tests.claims import Claim

# ---------------------------------------------------------------------------
# Symbolic Butcher order-condition machinery
# ---------------------------------------------------------------------------


def _butcher_conditions(
    A: list[list[sympy.Rational]],
    b: list[sympy.Rational],
    c: list[sympy.Rational],
    order: int,
) -> list[tuple[str, sympy.Expr]]:
    """Return (label, residual) pairs for all Butcher conditions up to order p.

    A residual of zero means the condition holds.  All arithmetic is exact
    when the inputs are sympy.Rational.

    Conditions up to p=5 are enumerated explicitly.  Phase 2 will replace
    these with the universal B-series framework (rooted-tree enumeration).
    """
    s = len(b)
    conditions: list[tuple[str, sympy.Expr]] = []

    conditions.append(("sum(b)=1", sum(b) - 1))

    if order >= 2:
        conditions.append(
            ("sum(b*c)=1/2", sum(b[i] * c[i] for i in range(s)) - sympy.Rational(1, 2))
        )

    if order >= 3:
        conditions.append(
            (
                "sum(b*c^2)=1/3",
                sum(b[i] * c[i] ** 2 for i in range(s)) - sympy.Rational(1, 3),
            )
        )
        conditions.append(
            (
                "sum(b*A*c)=1/6",
                sum(b[i] * sum(A[i][j] * c[j] for j in range(s)) for i in range(s))
                - sympy.Rational(1, 6),
            )
        )

    if order >= 4:
        conditions.append(
            (
                "sum(b*c^3)=1/4",
                sum(b[i] * c[i] ** 3 for i in range(s)) - sympy.Rational(1, 4),
            )
        )
        conditions.append(
            (
                "sum(b*c*A*c)=1/8",
                sum(
                    b[i] * c[i] * sum(A[i][j] * c[j] for j in range(s))
                    for i in range(s)
                )
                - sympy.Rational(1, 8),
            )
        )
        conditions.append(
            (
                "sum(b*A*c^2)=1/12",
                sum(b[i] * sum(A[i][j] * c[j] ** 2 for j in range(s)) for i in range(s))
                - sympy.Rational(1, 12),
            )
        )
        conditions.append(
            (
                "sum(b*A*A*c)=1/24",
                sum(
                    b[i]
                    * sum(
                        A[i][j] * sum(A[j][k] * c[k] for k in range(s))
                        for j in range(s)
                    )
                    for i in range(s)
                )
                - sympy.Rational(1, 24),
            )
        )

    if order >= 5:
        # 4 of the 17 p=5 conditions; Phase 2 adds the full set via B-series.
        conditions.append(
            (
                "sum(b*c^4)=1/5",
                sum(b[i] * c[i] ** 4 for i in range(s)) - sympy.Rational(1, 5),
            )
        )
        conditions.append(
            (
                "sum(b*c^2*A*c)=1/10",
                sum(
                    b[i] * c[i] ** 2 * sum(A[i][j] * c[j] for j in range(s))
                    for i in range(s)
                )
                - sympy.Rational(1, 10),
            )
        )
        conditions.append(
            (
                "sum(b*(A*c)^2)=1/20",
                sum(b[i] * sum(A[i][j] * c[j] for j in range(s)) ** 2 for i in range(s))
                - sympy.Rational(1, 20),
            )
        )
        conditions.append(
            (
                "sum(b*c*A*c^2)=1/15",
                sum(
                    b[i] * c[i] * sum(A[i][j] * c[j] ** 2 for j in range(s))
                    for i in range(s)
                )
                - sympy.Rational(1, 15),
            )
        )

    return conditions


# ---------------------------------------------------------------------------
# Claim classes
# ---------------------------------------------------------------------------

_DT_BASE = 0.1
_N_HALVINGS = 5


class _RKOrderClaim(Claim):
    """Tier A: verify Butcher order conditions hold symbolically."""

    def __init__(self, instance: RungeKuttaIntegrator, label: str) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"rk_order/{self._label}"

    def check(self) -> None:
        inst = self._instance
        conditions = _butcher_conditions(inst.A_sym, inst.b_sym, inst.c_sym, inst.order)
        for cond_label, residual in conditions:
            assert residual == 0, (
                f"{self._label}: Butcher condition '{cond_label}' failed; "
                f"residual = {residual}"
            )


class _ConvergenceClaim(Claim):
    """Tier B: verify convergence rate on dy/dt = λy."""

    def __init__(
        self,
        instance: RungeKuttaIntegrator,
        label: str,
        lam: float = -1.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._lam = lam

    @property
    def description(self) -> str:
        return f"rk_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam = self._lam
        rhs = BlackBoxRHS(lambda t, u, _lam=lam: _lam * u)

        dts = [_DT_BASE / (2**k) for k in range(_N_HALVINGS + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt)
            state = RKState(0.0, np.array([1.0]))
            for _ in range(n_steps):
                state = inst.step(rhs, state, dt)
            exact = math.exp(lam * state.t)
            errors.append(abs(float(state.u[0]) - exact))

        eps = np.finfo(float).eps * 10
        valid = [(dt, e) for dt, e in zip(dts, errors, strict=False) if e > eps]
        assert (
            len(valid) >= 3
        ), f"{self._label}: error reached machine precision too early"

        log_dts = [math.log(dt) for dt, _ in valid]
        log_errs = [math.log(e) for _, e in valid]
        n = len(log_dts)
        mean_x = sum(log_dts) / n
        mean_y = sum(log_errs) / n
        slope = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(log_dts, log_errs, strict=False)
        ) / sum((x - mean_x) ** 2 for x in log_dts)

        assert slope >= inst.order - 0.1, (
            f"{self._label}: convergence slope {slope:.3f} < declared order "
            f"{inst.order} - 0.1"
        )


class _StepperClaim(Claim):
    """Tier C: end-to-end TimeStepper.advance accuracy."""

    def __init__(
        self,
        instance: RungeKuttaIntegrator,
        label: str,
        dt: float,
        t_end: float = 1.0,
        lam: float = -1.0,
        rtol: float = 1e-4,
    ) -> None:
        self._instance = instance
        self._label = label
        self._dt = dt
        self._t_end = t_end
        self._lam = lam
        self._rtol = rtol

    @property
    def description(self) -> str:
        return f"stepper/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam = self._lam
        rhs = BlackBoxRHS(lambda t, u, _lam=lam: _lam * u)
        stepper = TimeStepper(inst, controller=ConstantStep(self._dt))

        final = stepper.advance(rhs, np.array([1.0]), 0.0, self._t_end)

        assert (
            abs(final.t - self._t_end) < 1e-12
        ), f"{self._label}: final time {final.t} != t_end {self._t_end}"
        exact = math.exp(lam * self._t_end)
        rel_err = abs(float(final.u[0]) - exact) / abs(exact)
        assert (
            rel_err < self._rtol
        ), f"{self._label}: relative error {rel_err:.2e} > rtol {self._rtol}"


# ---------------------------------------------------------------------------
# Claim registries
# ---------------------------------------------------------------------------

_ORDER_CLAIMS: list[_RKOrderClaim] = [
    _RKOrderClaim(forward_euler, "forward_euler"),
    _RKOrderClaim(midpoint, "midpoint"),
    _RKOrderClaim(heun, "heun"),
    _RKOrderClaim(ralston, "ralston"),
    _RKOrderClaim(rk4, "rk4"),
    _RKOrderClaim(dormand_prince, "dormand_prince"),
    _RKOrderClaim(bogacki_shampine, "bogacki_shampine"),
]

_CONVERGENCE_CLAIMS: list[_ConvergenceClaim] = [
    _ConvergenceClaim(forward_euler, "forward_euler"),
    _ConvergenceClaim(midpoint, "midpoint"),
    _ConvergenceClaim(heun, "heun"),
    _ConvergenceClaim(ralston, "ralston"),
    _ConvergenceClaim(rk4, "rk4"),
    _ConvergenceClaim(dormand_prince, "dormand_prince"),
    _ConvergenceClaim(bogacki_shampine, "bogacki_shampine"),
]

_STEPPER_CLAIMS: list[_StepperClaim] = [
    _StepperClaim(forward_euler, "forward_euler", dt=1e-3, rtol=1e-3),
    _StepperClaim(rk4, "rk4", dt=1e-2),
    _StepperClaim(dormand_prince, "dormand_prince", dt=1e-2),
]

# ---------------------------------------------------------------------------
# Parametric tests — one statement each, dispatching to claim.check()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    _ORDER_CLAIMS,
    ids=[c.description for c in _ORDER_CLAIMS],
)
def test_rk_order_conditions(claim: _RKOrderClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _CONVERGENCE_CLAIMS,
    ids=[c.description for c in _CONVERGENCE_CLAIMS],
)
def test_rk_convergence_rate(claim: _ConvergenceClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _STEPPER_CLAIMS,
    ids=[c.description for c in _STEPPER_CLAIMS],
)
def test_stepper_advance(claim: _StepperClaim) -> None:
    claim.check()
