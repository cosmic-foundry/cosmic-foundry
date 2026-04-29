"""Verification for the time-integration layer.

Claim classes and auto-discovery framework for TimeIntegrator subclasses:

  _RKOrderClaim          — B-series order conditions via rooted-tree enumeration
  _ConvergenceClaim      — temporal convergence rate on synthetic decay networks
  _StepperClaim          — end-to-end TimeStepper.advance accuracy
  _PIAccuracyClaim       — PIController achieves error ≤ c_rel · tol
  _PIWorkPrecisionClaim  — tighter tol yields smaller error

Each claim encodes both what is being verified and how to verify it.
Adding a new claim requires only appending to the relevant registry list;
the single parametric test per tier covers all entries.

Time-integrator verification is structurally distinct from spatial-operator
verification (different order conditions, different test problems, different
verification hierarchy), so this file is independent of test_convergence.py.
"""

from __future__ import annotations

import math
import sys

import pytest
import sympy

from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators import (
    ABState,
    AdamsBashforthIntegrator,
    AdditiveRHS,
    BlackBoxRHS,
    ConstantStep,
    CoxMatthewsETDRK4Integrator,
    DIRKIntegrator,
    ETDRK2Integrator,
    ExponentialEulerIntegrator,
    FamilySwitchingNordsieckIntegrator,
    IMEXIntegrator,
    JacobianRHS,
    KrogstadETDRK4Integrator,
    LinearPlusNonlinearRHS,
    NordsieckIntegrator,
    NordsieckState,
    OperatorSplitRHS,
    OrderSelector,
    PhiFunction,
    PIController,
    RKState,
    RungeKuttaIntegrator,
    SplittingStep,
    StiffnessDiagnostic,
    StiffnessSwitcher,
    StrangSplittingIntegrator,
    TimeStepper,
    VariableOrderNordsieckIntegrator,
    VODEController,
    ab2,
    ab3,
    ab4,
    adams_family,
    adams_moulton1,
    adams_moulton2,
    adams_moulton3,
    adams_moulton4,
    ars222,
    backward_euler,
    bdf1,
    bdf2,
    bdf3,
    bdf4,
    bdf_family,
    bogacki_shampine,
    cox_matthews_etdrk4,
    crouzeix_3,
    dormand_prince,
    elementary_weight,
    etd_euler,
    etdrk2,
    forward_euler,
    gamma,
    heun,
    implicit_midpoint,
    krogstad_etdrk4,
    lie_steps,
    midpoint,
    nordsieck_solution_distance,
    ralston,
    rk4,
    stability_function,
    strang_steps,
    trees_up_to_order,
)
from tests.claims import Claim

# ---------------------------------------------------------------------------
# Claim classes
# ---------------------------------------------------------------------------

_DT_BASE = 0.1
_N_HALVINGS = 5

# Synthetic thermonuclear-shaped baseline problem.
#
# A two-species closed capture/decay chain X0 -> X1 is the smallest
# mass-fraction network with the invariants we care about for astrophysical
# ODE integration: positivity, exact abundance conservation, a sparse
# zero-column-sum Jacobian, and an analytical solution for convergence claims.
_BASE_K = 1.0


def _base_network_f(t: float, u: Tensor) -> Tensor:
    x0 = float(u[0])
    return Tensor([-_BASE_K * x0, _BASE_K * x0], backend=u.backend)


def _base_network_jac(t: float, u: Tensor) -> Tensor:
    return Tensor([[-_BASE_K, 0.0], [_BASE_K, 0.0]], backend=u.backend)


_BASE_RHS = JacobianRHS(f=_base_network_f, jac=_base_network_jac)


def _base_network_exact(t: float) -> tuple[float, float]:
    x0 = math.exp(-_BASE_K * t)
    return x0, 1.0 - x0


def _max_base_network_error(u: Tensor, t: float) -> float:
    exact = _base_network_exact(t)
    return max(abs(float(u[i]) - exact[i]) for i in range(2))


def _assert_mass_fractions(
    u: Tensor,
    *,
    label: str,
    n: int,
    conservation_tol: float = 1e-12,
) -> None:
    total = sum(float(u[i]) for i in range(n))
    assert abs(total - 1.0) < conservation_tol, (
        f"{label}: sum(X) = {total:.15f} != 1 "
        f"(mass not conserved to {conservation_tol:.1e})"
    )
    for i in range(n):
        xi = float(u[i])
        assert xi >= -1e-10, f"{label}: X[{i}] = {xi:.3e} < 0"


class _RKOrderClaim(Claim):
    """Verify B-series order conditions: α(τ) = 1/γ(τ) for all trees with |τ| ≤ p."""

    def __init__(
        self, instance: RungeKuttaIntegrator | DIRKIntegrator, label: str
    ) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"rk_order/{self._label}"

    def check(self) -> None:
        inst = self._instance
        for t in trees_up_to_order(inst.order):
            alpha = elementary_weight(t, inst.A_sym, inst.b_sym)
            expected = sympy.Rational(1) / gamma(t)
            assert sympy.simplify(alpha - expected) == 0, (
                f"{self._label}: B-series condition failed for tree {t}; "
                f"α(τ)={alpha}, 1/γ(τ)={expected}"
            )


class _ConvergenceClaim(Claim):
    """Tier B: verify convergence rate on a closed three-species network."""

    def __init__(
        self,
        instance: RungeKuttaIntegrator,
        label: str,
    ) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"rk_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        rhs = BlackBoxRHS(_decay_f)

        dts = [_DT_BASE / (2**k) for k in range(_N_HALVINGS + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt)
            state = RKState(0.0, Tensor([1.0, 0.0, 0.0]))
            for _ in range(n_steps):
                state = inst.step(rhs, state, dt)
            errors.append(_max_decay_error(state.u, state.t))

        eps = sys.float_info.epsilon * 10
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
        rtol: float = 1e-4,
    ) -> None:
        self._instance = instance
        self._label = label
        self._dt = dt
        self._t_end = t_end
        self._rtol = rtol

    @property
    def description(self) -> str:
        return f"stepper/{self._label}"

    def check(self) -> None:
        inst = self._instance
        rhs = BlackBoxRHS(_base_network_f)
        stepper = TimeStepper(inst, controller=ConstantStep(self._dt))

        final = stepper.advance(rhs, Tensor([1.0, 0.0]), 0.0, self._t_end)

        assert (
            abs(final.t - self._t_end) < 1e-12
        ), f"{self._label}: final time {final.t} != t_end {self._t_end}"
        rel_err = _max_base_network_error(final.u, self._t_end)
        assert (
            rel_err < self._rtol
        ), f"{self._label}: relative error {rel_err:.2e} > rtol {self._rtol}"
        _assert_mass_fractions(final.u, label=self._label, n=2)


# ---------------------------------------------------------------------------
# Phase 1 claims — PIController
# ---------------------------------------------------------------------------

# Default PI exponents (Hairer Vol. II recommendation): α = 0.7/p, β = 0.4/p.
# These give a well-damped response for explicit RK methods on smooth problems.
_PI_ALPHA = 0.7
_PI_BETA = 0.4


class _PIAccuracyClaim(Claim):
    """Verify that PIController achieves error ≤ c_rel · tol on a decay network.

    Integrates the closed two-species mass-fraction network from 0 to t_end
    with a PIController.  The c_rel factor absorbs pre-asymptotic constants;
    10 is generous but distinguishes correct control from uncontrolled growth.
    """

    def __init__(
        self,
        instance: RungeKuttaIntegrator,
        label: str,
        tol: float = 1e-4,
        t_end: float = 1.0,
        c_rel: float = 10.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._tol = tol
        self._t_end = t_end
        self._c_rel = c_rel

    @property
    def description(self) -> str:
        return f"pi_accuracy/{self._label}"

    def check(self) -> None:
        inst = self._instance
        p = inst.order
        rhs = BlackBoxRHS(_base_network_f)
        pi = PIController(
            alpha=_PI_ALPHA / p,
            beta=_PI_BETA / p,
            tol=self._tol,
            dt0=0.1,
        )
        stepper = TimeStepper(inst, controller=pi)
        final = stepper.advance(rhs, Tensor([1.0, 0.0]), 0.0, self._t_end)
        err = _max_base_network_error(final.u, self._t_end)
        assert (
            err <= self._c_rel * self._tol
        ), f"{self._label}: error {err:.2e} > {self._c_rel} × tol {self._tol:.2e}"
        _assert_mass_fractions(final.u, label=self._label, n=2)


class _PIWorkPrecisionClaim(Claim):
    """Verify that error decreases as tol decreases (adaptive convergence).

    Runs the integrator at two tolerances differing by a factor of 10 and
    checks that the error at the tighter tolerance is smaller.  This is the
    minimal falsifiable claim that adaptive control is working: if the
    controller were ignoring tol, both runs would give the same error.
    """

    def __init__(
        self,
        instance: RungeKuttaIntegrator,
        label: str,
        tol_coarse: float = 1e-3,
        tol_fine: float = 1e-5,
        t_end: float = 1.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._tol_coarse = tol_coarse
        self._tol_fine = tol_fine
        self._t_end = t_end

    @property
    def description(self) -> str:
        return f"pi_work_precision/{self._label}"

    def check(self) -> None:
        inst = self._instance
        p = inst.order
        rhs = BlackBoxRHS(_base_network_f)

        def _run(tol: float) -> float:
            pi = PIController(
                alpha=_PI_ALPHA / p,
                beta=_PI_BETA / p,
                tol=tol,
                dt0=0.1,
            )
            final = TimeStepper(inst, controller=pi).advance(
                rhs, Tensor([1.0, 0.0]), 0.0, self._t_end
            )
            _assert_mass_fractions(final.u, label=self._label, n=2)
            return _max_base_network_error(final.u, self._t_end)

        err_coarse = _run(self._tol_coarse)
        err_fine = _run(self._tol_fine)
        assert err_fine < err_coarse, (
            f"{self._label}: tighter tol did not reduce error "
            f"(coarse={err_coarse:.2e}, fine={err_fine:.2e})"
        )


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
    _RKOrderClaim(backward_euler, "backward_euler"),
    _RKOrderClaim(implicit_midpoint, "implicit_midpoint"),
    _RKOrderClaim(crouzeix_3, "crouzeix_3"),
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

_PI_ACCURACY_CLAIMS: list[_PIAccuracyClaim] = [
    _PIAccuracyClaim(heun, "heun"),
    _PIAccuracyClaim(bogacki_shampine, "bogacki_shampine"),
    _PIAccuracyClaim(dormand_prince, "dormand_prince"),
]

_PI_WORK_PRECISION_CLAIMS: list[_PIWorkPrecisionClaim] = [
    _PIWorkPrecisionClaim(heun, "heun"),
    _PIWorkPrecisionClaim(bogacki_shampine, "bogacki_shampine"),
    _PIWorkPrecisionClaim(dormand_prince, "dormand_prince"),
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
    _PI_ACCURACY_CLAIMS,
    ids=[c.description for c in _PI_ACCURACY_CLAIMS],
)
def test_pi_accuracy(claim: _PIAccuracyClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _PI_WORK_PRECISION_CLAIMS,
    ids=[c.description for c in _PI_WORK_PRECISION_CLAIMS],
)
def test_pi_work_precision(claim: _PIWorkPrecisionClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _STEPPER_CLAIMS,
    ids=[c.description for c in _STEPPER_CLAIMS],
)
def test_stepper_advance(claim: _StepperClaim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 4 claims — DIRK integrators
# ---------------------------------------------------------------------------


class _DIRKConvergenceClaim(Claim):
    """Verify convergence order on the baseline abundance network."""

    def __init__(
        self,
        instance: DIRKIntegrator,
        label: str,
    ) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"dirk_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        rhs = _BASE_RHS
        dts = [_DT_BASE / (2**k) for k in range(_N_HALVINGS + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt)
            state = RKState(0.0, Tensor([1.0, 0.0]))
            for _ in range(n_steps):
                state = inst.step(rhs, state, dt)
            errors.append(_max_base_network_error(state.u, state.t))

        eps = sys.float_info.epsilon * 10
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


class _AStabilityClaim(Claim):
    """Verify A-stability: |R(iω)| ≤ 1 for sampled imaginary-axis points."""

    def __init__(self, instance: DIRKIntegrator, label: str) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"a_stability/{self._label}"

    def check(self) -> None:
        inst = self._instance
        R_expr = stability_function(inst.A_sym, inst.b_sym)
        z = sympy.Symbol("z")
        R_func = sympy.lambdify(z, R_expr, modules="cmath")
        for omega in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            R_val = abs(R_func(complex(0.0, omega)))
            assert (
                R_val <= 1.0 + 1e-10
            ), f"{self._label}: |R(i·{omega})| = {R_val:.8f} > 1 (A-stability violated)"


class _LStabilityClaim(Claim):
    """Verify L-stability: |R(z)| → 0 as Re(z) → −∞."""

    def __init__(self, instance: DIRKIntegrator, label: str) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"l_stability/{self._label}"

    def check(self) -> None:
        inst = self._instance
        R_expr = stability_function(inst.A_sym, inst.b_sym)
        z = sympy.Symbol("z")
        R_func = sympy.lambdify(z, R_expr, modules="cmath")
        R_inf = abs(R_func(-1e6))
        assert (
            R_inf < 1e-4
        ), f"{self._label}: |R(-1e6)| = {R_inf:.2e}; method is not L-stable"


_DIRK_CONVERGENCE_CLAIMS: list[_DIRKConvergenceClaim] = [
    _DIRKConvergenceClaim(backward_euler, "backward_euler"),
    _DIRKConvergenceClaim(implicit_midpoint, "implicit_midpoint"),
    _DIRKConvergenceClaim(crouzeix_3, "crouzeix_3"),
]

_A_STABILITY_CLAIMS: list[_AStabilityClaim] = [
    _AStabilityClaim(backward_euler, "backward_euler"),
    _AStabilityClaim(implicit_midpoint, "implicit_midpoint"),
    _AStabilityClaim(crouzeix_3, "crouzeix_3"),
]

_L_STABILITY_CLAIMS: list[_LStabilityClaim] = [
    _LStabilityClaim(backward_euler, "backward_euler"),
]


@pytest.mark.parametrize(
    "claim",
    _DIRK_CONVERGENCE_CLAIMS,
    ids=[c.description for c in _DIRK_CONVERGENCE_CLAIMS],
)
def test_dirk_convergence(claim: _DIRKConvergenceClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _A_STABILITY_CLAIMS,
    ids=[c.description for c in _A_STABILITY_CLAIMS],
)
def test_a_stability(claim: _AStabilityClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _L_STABILITY_CLAIMS,
    ids=[c.description for c in _L_STABILITY_CLAIMS],
)
def test_l_stability(claim: _LStabilityClaim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 5 claims — IMEX + abundance conservation
# ---------------------------------------------------------------------------

# Canonical test problem: 3-species closed decay chain A→B→C with rates k1=1, k2=2.
#
# This is the minimal nuclear-astrophysics-shaped ODE: the state vector X = [X0,X1,X2]
# represents mass fractions with X_i ∈ [0,1] and sum(X_i) = 1.  The rate matrix
# has zero column sums, so any Runge-Kutta method (explicit or DIRK or IMEX) preserves
# sum(X) exactly in floating point — making conservation a hard check, not a tolerance.
#
# Exact solution starting from X(0) = [1, 0, 0]:
#   X0(t) = exp(−t)
#   X1(t) = exp(−t) − exp(−2t)
#   X2(t) = 1 − X0 − X1
#
# IMEX split: f_I handles self-decay (diagonal, stiff in principle), f_E handles
# inter-species production (off-diagonal, always non-stiff).  This mirrors the
# standard operator split in nuclear burning codes.

_K1 = 1.0
_K2 = 2.0

# Total RHS for explicit/DIRK use (via JacobianRHS which also satisfies RHSProtocol).
_DECAY_JAC = [[-_K1, 0.0, 0.0], [_K1, -_K2, 0.0], [0.0, _K2, 0.0]]


def _decay_f(t: float, u: Tensor) -> Tensor:
    x0, x1 = float(u[0]), float(u[1])
    return Tensor([-_K1 * x0, _K1 * x0 - _K2 * x1, _K2 * x1], backend=u.backend)


def _decay_jac(t: float, u: Tensor) -> Tensor:
    return Tensor(_DECAY_JAC, backend=u.backend)


_DECAY_RHS = JacobianRHS(f=_decay_f, jac=_decay_jac)

# IMEX split: f_I = diagonal decay, f_E = off-diagonal production.
_DECAY_RHS_IMEX = AdditiveRHS(
    f_E=lambda t, u: Tensor(
        [0.0, _K1 * float(u[0]), _K2 * float(u[1])], backend=u.backend
    ),
    f_I=lambda t, u: Tensor(
        [-_K1 * float(u[0]), -_K2 * float(u[1]), 0.0], backend=u.backend
    ),
    jac_I=lambda t, u: Tensor(
        [[-_K1, 0.0, 0.0], [0.0, -_K2, 0.0], [0.0, 0.0, 0.0]], backend=u.backend
    ),
)


def _decay_exact(t: float) -> tuple[float, float, float]:
    x0 = math.exp(-_K1 * t)
    x1 = math.exp(-_K1 * t) - math.exp(-_K2 * t)
    return x0, x1, 1.0 - x0 - x1


def _max_decay_error(u: Tensor, t: float) -> float:
    exact = _decay_exact(t)
    return max(abs(float(u[i]) - exact[i]) for i in range(3))


class _AbundanceConservationClaim(Claim):
    """Verify a closed decay chain: accuracy, mass-fraction sum = 1, positivity.

    Applies to any integrator whose step(rhs, state, dt) → RKState interface
    accepts a JacobianRHS (which also satisfies the plain RHSProtocol, so
    explicit RK methods work too).
    """

    def __init__(
        self,
        instance: RungeKuttaIntegrator | DIRKIntegrator,
        label: str,
        dt: float = 0.05,
        t_end: float = 2.0,
        acc_tol: float = 1e-3,
    ) -> None:
        self._instance = instance
        self._label = label
        self._dt = dt
        self._t_end = t_end
        self._acc_tol = acc_tol

    @property
    def description(self) -> str:
        return f"abundance_conservation/{self._label}"

    def check(self) -> None:
        inst = self._instance
        n_steps = round(self._t_end / self._dt)
        state = RKState(0.0, Tensor([1.0, 0.0, 0.0]))
        for _ in range(n_steps):
            state = inst.step(_DECAY_RHS, state, self._dt)

        x0_ex, x1_ex, x2_ex = _decay_exact(state.t)
        err = max(
            abs(float(state.u[0]) - x0_ex),
            abs(float(state.u[1]) - x1_ex),
            abs(float(state.u[2]) - x2_ex),
        )
        assert (
            err < self._acc_tol
        ), f"{self._label}: max abundance error {err:.2e} > {self._acc_tol:.2e}"

        total = sum(float(state.u[i]) for i in range(3))
        assert (
            abs(total - 1.0) < 1e-12
        ), f"{self._label}: sum(X) = {total:.15f} ≠ 1 (mass not conserved)"

        for i in range(3):
            xi = float(state.u[i])
            assert (
                xi >= -1e-10
            ), f"{self._label}: X[{i}] = {xi:.3e} < 0 (positivity violated)"


class _IMEXAbundanceConservationClaim(Claim):
    """Same decay-chain checks for an IMEXIntegrator with the production/decay split."""

    def __init__(
        self,
        instance: IMEXIntegrator,
        label: str,
        dt: float = 0.05,
        t_end: float = 2.0,
        acc_tol: float = 1e-3,
    ) -> None:
        self._instance = instance
        self._label = label
        self._dt = dt
        self._t_end = t_end
        self._acc_tol = acc_tol

    @property
    def description(self) -> str:
        return f"imex_abundance_conservation/{self._label}"

    def check(self) -> None:
        inst = self._instance
        n_steps = round(self._t_end / self._dt)
        state = RKState(0.0, Tensor([1.0, 0.0, 0.0]))
        for _ in range(n_steps):
            state = inst.step(_DECAY_RHS_IMEX, state, self._dt)

        x0_ex, x1_ex, x2_ex = _decay_exact(state.t)
        err = max(
            abs(float(state.u[0]) - x0_ex),
            abs(float(state.u[1]) - x1_ex),
            abs(float(state.u[2]) - x2_ex),
        )
        assert (
            err < self._acc_tol
        ), f"{self._label}: max abundance error {err:.2e} > {self._acc_tol:.2e}"

        total = sum(float(state.u[i]) for i in range(3))
        assert (
            abs(total - 1.0) < 1e-12
        ), f"{self._label}: sum(X) = {total:.15f} ≠ 1 (mass not conserved)"

        for i in range(3):
            xi = float(state.u[i])
            assert (
                xi >= -1e-10
            ), f"{self._label}: X[{i}] = {xi:.3e} < 0 (positivity violated)"


class _IMEXConvergenceClaim(Claim):
    """Verify IMEX convergence order on the baseline abundance network."""

    def __init__(
        self,
        instance: IMEXIntegrator,
        label: str,
    ) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"imex_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        rhs = AdditiveRHS(
            f_E=lambda t, u: Tensor([0.0, _BASE_K * float(u[0])], backend=u.backend),
            f_I=lambda t, u: Tensor([-_BASE_K * float(u[0]), 0.0], backend=u.backend),
            jac_I=lambda t, u: Tensor([[-_BASE_K, 0.0], [0.0, 0.0]], backend=u.backend),
        )
        dts = [_DT_BASE / (2**k) for k in range(_N_HALVINGS + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt)
            state = RKState(0.0, Tensor([1.0, 0.0]))
            for _ in range(n_steps):
                state = inst.step(rhs, state, dt)
            errors.append(_max_base_network_error(state.u, state.t))

        eps = sys.float_info.epsilon * 10
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


_ABUNDANCE_CONSERVATION_CLAIMS: list[_AbundanceConservationClaim] = [
    _AbundanceConservationClaim(rk4, "rk4", acc_tol=1e-4),
    _AbundanceConservationClaim(backward_euler, "backward_euler", acc_tol=2e-2),
    _AbundanceConservationClaim(implicit_midpoint, "implicit_midpoint", acc_tol=1e-3),
    _AbundanceConservationClaim(crouzeix_3, "crouzeix_3", acc_tol=5e-4),
]

_IMEX_ABUNDANCE_CONSERVATION_CLAIMS: list[_IMEXAbundanceConservationClaim] = [
    _IMEXAbundanceConservationClaim(ars222, "ars222"),
]

_IMEX_CONVERGENCE_CLAIMS: list[_IMEXConvergenceClaim] = [
    _IMEXConvergenceClaim(ars222, "ars222"),
]


@pytest.mark.parametrize(
    "claim",
    _ABUNDANCE_CONSERVATION_CLAIMS,
    ids=[c.description for c in _ABUNDANCE_CONSERVATION_CLAIMS],
)
def test_abundance_conservation(claim: _AbundanceConservationClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _IMEX_ABUNDANCE_CONSERVATION_CLAIMS,
    ids=[c.description for c in _IMEX_ABUNDANCE_CONSERVATION_CLAIMS],
)
def test_imex_abundance_conservation(claim: _IMEXAbundanceConservationClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _IMEX_CONVERGENCE_CLAIMS,
    ids=[c.description for c in _IMEX_CONVERGENCE_CLAIMS],
)
def test_imex_convergence(claim: _IMEXConvergenceClaim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 6 claims — Adams-Bashforth multistep
# ---------------------------------------------------------------------------


class _ABConvergenceClaim(Claim):
    """Verify Adams-Bashforth convergence rate on the baseline abundance network.

    The first k−1 steps are bootstrapped with RK4 internally; the slope
    is measured across the full dt grid and reflects the AB order because
    the RK4 bootstrap error is the same order as AB4 and higher for AB2/AB3.
    """

    def __init__(
        self,
        instance: AdamsBashforthIntegrator,
        label: str,
    ) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"ab_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        rhs = BlackBoxRHS(_base_network_f)
        dts = [_DT_BASE / (2**k) for k in range(_N_HALVINGS + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt)
            state = ABState(0.0, Tensor([1.0, 0.0]))
            for _ in range(n_steps):
                state = inst.step(rhs, state, dt)
            errors.append(_max_base_network_error(state.u, state.t))

        eps = sys.float_info.epsilon * 10
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


class _ABAbundanceConservationClaim(Claim):
    """Decay-chain abundance checks for Adams-Bashforth integrators.

    Uses the same 3-species A→B→C problem as the DIRK/IMEX abundance claims.
    JacobianRHS satisfies RHSProtocol (plain __call__), so it works with the
    AB interface without exposing the Jacobian to the integrator.
    """

    def __init__(
        self,
        instance: AdamsBashforthIntegrator,
        label: str,
        dt: float = 0.05,
        t_end: float = 2.0,
        acc_tol: float = 1e-3,
    ) -> None:
        self._instance = instance
        self._label = label
        self._dt = dt
        self._t_end = t_end
        self._acc_tol = acc_tol

    @property
    def description(self) -> str:
        return f"ab_abundance_conservation/{self._label}"

    def check(self) -> None:
        inst = self._instance
        n_steps = round(self._t_end / self._dt)
        state = ABState(0.0, Tensor([1.0, 0.0, 0.0]))
        for _ in range(n_steps):
            state = inst.step(_DECAY_RHS, state, self._dt)

        x0_ex, x1_ex, x2_ex = _decay_exact(state.t)
        err = max(
            abs(float(state.u[0]) - x0_ex),
            abs(float(state.u[1]) - x1_ex),
            abs(float(state.u[2]) - x2_ex),
        )
        assert (
            err < self._acc_tol
        ), f"{self._label}: max abundance error {err:.2e} > {self._acc_tol:.2e}"

        total = sum(float(state.u[i]) for i in range(3))
        assert (
            abs(total - 1.0) < 1e-12
        ), f"{self._label}: sum(X) = {total:.15f} ≠ 1 (mass not conserved)"

        for i in range(3):
            xi = float(state.u[i])
            assert (
                xi >= -1e-10
            ), f"{self._label}: X[{i}] = {xi:.3e} < 0 (positivity violated)"


_AB_CONVERGENCE_CLAIMS: list[_ABConvergenceClaim] = [
    _ABConvergenceClaim(ab2, "ab2"),
    _ABConvergenceClaim(ab3, "ab3"),
    _ABConvergenceClaim(ab4, "ab4"),
]

_AB_ABUNDANCE_CONSERVATION_CLAIMS: list[_ABAbundanceConservationClaim] = [
    _ABAbundanceConservationClaim(ab2, "ab2", acc_tol=5e-4),
    _ABAbundanceConservationClaim(ab3, "ab3", acc_tol=5e-5),
    _ABAbundanceConservationClaim(ab4, "ab4", acc_tol=5e-6),
]


@pytest.mark.parametrize(
    "claim",
    _AB_CONVERGENCE_CLAIMS,
    ids=[c.description for c in _AB_CONVERGENCE_CLAIMS],
)
def test_ab_convergence(claim: _ABConvergenceClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _AB_ABUNDANCE_CONSERVATION_CLAIMS,
    ids=[c.description for c in _AB_ABUNDANCE_CONSERVATION_CLAIMS],
)
def test_ab_abundance_conservation(claim: _ABAbundanceConservationClaim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 7a claims — Nordsieck BDF fixed-order integrators
# ---------------------------------------------------------------------------


_BDF_N_HALVINGS = 7  # extra halvings vs AB to clear bootstrap pre-asymptotic transient


class _BDFConvergenceClaim(Claim):
    """Verify BDF-q convergence rate on the baseline abundance network.

    Bootstraps q RK4 steps then measures the log-log slope of final-time
    error vs step size.  Slope ≥ declared order − 0.1 is required.

    Step-size range is chosen to keep the regression entirely in the
    asymptotic convergence regime:

    - Base step _DT_BASE / q: the bootstrap takes q RK4 steps, and the
      Nordsieck correction needs roughly q more BDF steps to propagate
      through all history slots, so the number of post-bootstrap steps
      must grow with q.  Dividing the base by q guarantees ~9q steps at
      the coarsest dt, which is sufficient for any order tested here.

    - Halvings _BDF_N_HALVINGS − (floor(log₂ q) + 1): reducing dt_base
      by q moves the fine end q times closer to machine precision.  Since
      the error scales as dt^q, each halving buys a factor 2^q in
      accuracy, so the useful range shrinks by log₂ q halvings.  The
      extra −1 keeps the finest error safely above the noise floor
      (roughly 2000× machine epsilon rather than ~15×).

    The RHS is a JacobianRHS so both the BDF Newton corrector and the
    plain __call__ are exercised.
    """

    def __init__(
        self,
        instance: NordsieckIntegrator,
        label: str,
    ) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"bdf_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        rhs = _BASE_RHS
        dt_base = _DT_BASE / inst.order
        n_halvings = _BDF_N_HALVINGS - (math.floor(math.log2(inst.order)) + 1)
        dts = [dt_base / (2**k) for k in range(n_halvings + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt) - inst.order
            state = inst.init_state(rhs, 0.0, Tensor([1.0, 0.0]), dt)
            for _ in range(max(n_steps, 1)):
                state = inst.step(rhs, state, dt)
            errors.append(_max_base_network_error(state.u, state.t))

        eps = sys.float_info.epsilon * 10
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
            f"{inst.order} − 0.1"
        )


class _BDFConservationClaim(Claim):
    """Verify BDF-q on the 3-species A→B→C decay chain.

    Uses the same _DECAY_RHS as the DIRK and IMEX conservation claims.
    Checks accuracy against the analytical solution, exact mass conservation
    |ΣXᵢ − 1| < 1e-12, and positivity.  Conservation is exact because
    f has zero column sums and Newton converges in one step for linear f.
    """

    def __init__(
        self,
        instance: NordsieckIntegrator,
        label: str,
        dt: float = 0.05,
        t_end: float = 2.0,
        acc_tol: float = 1e-3,
    ) -> None:
        self._instance = instance
        self._label = label
        self._dt = dt
        self._t_end = t_end
        self._acc_tol = acc_tol

    @property
    def description(self) -> str:
        return f"bdf_conservation/{self._label}"

    def check(self) -> None:
        inst = self._instance
        dt = self._dt
        n_steps = round(self._t_end / dt) - inst.order
        state = inst.init_state(_DECAY_RHS, 0.0, Tensor([1.0, 0.0, 0.0]), dt)
        for _ in range(max(n_steps, 1)):
            state = inst.step(_DECAY_RHS, state, dt)

        x0_ex, x1_ex, x2_ex = _decay_exact(state.t)
        err = max(
            abs(float(state.u[0]) - x0_ex),
            abs(float(state.u[1]) - x1_ex),
            abs(float(state.u[2]) - x2_ex),
        )
        assert (
            err < self._acc_tol
        ), f"{self._label}: max abundance error {err:.2e} > {self._acc_tol:.2e}"

        total = sum(float(state.u[i]) for i in range(3))
        assert (
            abs(total - 1.0) < 1e-12
        ), f"{self._label}: sum(X) = {total:.15f} ≠ 1 (mass not conserved)"

        for i in range(3):
            xi = float(state.u[i])
            assert (
                xi >= -1e-10
            ), f"{self._label}: X[{i}] = {xi:.3e} < 0 (positivity violated)"


_BDF_CONVERGENCE_CLAIMS: list[_BDFConvergenceClaim] = [
    _BDFConvergenceClaim(bdf1, "bdf1"),
    _BDFConvergenceClaim(bdf2, "bdf2"),
    _BDFConvergenceClaim(bdf3, "bdf3"),
    _BDFConvergenceClaim(bdf4, "bdf4"),
]

_BDF_CONSERVATION_CLAIMS: list[_BDFConservationClaim] = [
    _BDFConservationClaim(bdf1, "bdf1", acc_tol=0.1),
    _BDFConservationClaim(bdf2, "bdf2", acc_tol=5e-3),
    _BDFConservationClaim(bdf3, "bdf3", acc_tol=5e-4),
    _BDFConservationClaim(bdf4, "bdf4", acc_tol=5e-5),
]


@pytest.mark.parametrize(
    "claim",
    _BDF_CONVERGENCE_CLAIMS,
    ids=[c.description for c in _BDF_CONVERGENCE_CLAIMS],
)
def test_bdf_convergence(claim: _BDFConvergenceClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _BDF_CONSERVATION_CLAIMS,
    ids=[c.description for c in _BDF_CONSERVATION_CLAIMS],
)
def test_bdf_conservation(claim: _BDFConservationClaim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 8 claims — Adams-Moulton (Nordsieck form, fixed-point corrector)
# ---------------------------------------------------------------------------

_AM_N_HALVINGS = _BDF_N_HALVINGS  # same bootstrap depth as BDF


class _AMConvergenceClaim(Claim):
    """Verify Adams-Moulton-q convergence rate on the baseline abundance network.

    Mirrors _BDFConvergenceClaim exactly, but uses plain RHSProtocol
    (no Jacobian) because Adams methods use fixed-point iteration.

    Step-size calibration follows the same logic as BDF:

    - Base step _DT_BASE / q: the bootstrap takes q RK4 steps, and the
      Nordsieck correction needs roughly q more AM steps to propagate
      through all history slots, so the coarsest dt must give ~9q steps.

    - Halvings _AM_N_HALVINGS − (floor(log₂ q) + 1): same argument as
      BDF — reducing dt_base by q moves the fine end closer to machine
      precision by log₂ q halvings.
    """

    def __init__(
        self,
        instance: NordsieckIntegrator,
        label: str,
    ) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"am_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        rhs = BlackBoxRHS(_base_network_f)
        dt_base = _DT_BASE / inst.order
        n_halvings = _AM_N_HALVINGS - (math.floor(math.log2(inst.order)) + 1)
        dts = [dt_base / (2**k) for k in range(n_halvings + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt) - inst.order
            state = inst.init_state(rhs, 0.0, Tensor([1.0, 0.0]), dt)
            for _ in range(max(n_steps, 1)):
                state = inst.step(rhs, state, dt)
            errors.append(_max_base_network_error(state.u, state.t))

        eps = sys.float_info.epsilon * 10
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
            f"{inst.order} − 0.1"
        )


class _AMConservationClaim(Claim):
    """Verify Adams-Moulton-q on the 3-species A→B→C decay chain.

    Uses _decay_f directly (plain RHSProtocol — no Jacobian required).
    The rate matrix has zero column sums, so fixed-point iteration
    preserves sum(X) to machine precision.  Checks accuracy against the
    analytical solution, |ΣXᵢ − 1| < 1e-12, and positivity.
    """

    def __init__(
        self,
        instance: NordsieckIntegrator,
        label: str,
        dt: float = 0.05,
        t_end: float = 2.0,
        acc_tol: float = 1e-3,
    ) -> None:
        self._instance = instance
        self._label = label
        self._dt = dt
        self._t_end = t_end
        self._acc_tol = acc_tol

    @property
    def description(self) -> str:
        return f"am_conservation/{self._label}"

    def check(self) -> None:
        inst = self._instance
        dt = self._dt
        n_steps = round(self._t_end / dt) - inst.order
        state = inst.init_state(_decay_f, 0.0, Tensor([1.0, 0.0, 0.0]), dt)
        for _ in range(max(n_steps, 1)):
            state = inst.step(_decay_f, state, dt)

        x0_ex, x1_ex, x2_ex = _decay_exact(state.t)
        err = max(
            abs(float(state.u[0]) - x0_ex),
            abs(float(state.u[1]) - x1_ex),
            abs(float(state.u[2]) - x2_ex),
        )
        assert (
            err < self._acc_tol
        ), f"{self._label}: max abundance error {err:.2e} > {self._acc_tol:.2e}"

        total = sum(float(state.u[i]) for i in range(3))
        assert (
            abs(total - 1.0) < 1e-12
        ), f"{self._label}: sum(X) = {total:.15f} ≠ 1 (mass not conserved)"

        for i in range(3):
            xi = float(state.u[i])
            assert (
                xi >= -1e-10
            ), f"{self._label}: X[{i}] = {xi:.3e} < 0 (positivity violated)"


_AM_CONVERGENCE_CLAIMS: list[_AMConvergenceClaim] = [
    _AMConvergenceClaim(adams_moulton1, "am1"),
    _AMConvergenceClaim(adams_moulton2, "am2"),
    _AMConvergenceClaim(adams_moulton3, "am3"),
    _AMConvergenceClaim(adams_moulton4, "am4"),
]

_AM_CONSERVATION_CLAIMS: list[_AMConservationClaim] = [
    _AMConservationClaim(adams_moulton1, "am1", acc_tol=0.05),
    _AMConservationClaim(adams_moulton2, "am2", acc_tol=5e-4),
    _AMConservationClaim(adams_moulton3, "am3", acc_tol=1e-5),
    _AMConservationClaim(adams_moulton4, "am4", acc_tol=1e-6),
]


@pytest.mark.parametrize(
    "claim",
    _AM_CONVERGENCE_CLAIMS,
    ids=[c.description for c in _AM_CONVERGENCE_CLAIMS],
)
def test_am_convergence(claim: _AMConvergenceClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _AM_CONSERVATION_CLAIMS,
    ids=[c.description for c in _AM_CONSERVATION_CLAIMS],
)
def test_am_conservation(claim: _AMConservationClaim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 9 claims — Nordsieck order-change and step-size rescaling
# ---------------------------------------------------------------------------


class _NordsieckRoundTripClaim(Claim):
    """Verify that pure Nordsieck transformations preserve the represented state."""

    @property
    def description(self) -> str:
        return "nordsieck_round_trip/order_and_step_size"

    def check(self) -> None:
        state = NordsieckState(
            t=0.25,
            h=0.1,
            z=(
                Tensor([0.8, 0.2]),
                Tensor([-0.08, 0.08]),
                Tensor([0.004, -0.004]),
            ),
        )

        raised = state.change_order(4)
        lowered = raised.change_order(2)
        assert lowered.q == state.q
        assert lowered.t == state.t
        assert lowered.h == state.h
        for lhs, rhs in zip(lowered.z, state.z, strict=True):
            assert float(norm(lhs - rhs)) == 0.0

        assert raised.q == 4
        assert float(norm(raised.z[0] - state.z[0])) == 0.0
        assert float(norm(raised.z[3])) == 0.0
        assert float(norm(raised.z[4])) == 0.0

        rescaled = state.rescale_step(0.05).rescale_step(0.1)
        assert rescaled.h == state.h
        for lhs, rhs in zip(rescaled.z, state.z, strict=True):
            assert float(norm(lhs - rhs)) < 1e-16


class _NordsieckRescaledAccuracyClaim(Claim):
    """Verify transformed Nordsieck states retain fixed-order accuracy."""

    def __init__(
        self,
        instance: NordsieckIntegrator,
        label: str,
        rhs: BlackBoxRHS | JacobianRHS,
        q_source: int = 4,
        q_target: int = 2,
        h_source: float = 0.1,
        h_target: float = 0.025,
    ) -> None:
        self._instance = instance
        self._label = label
        self._rhs = rhs
        self._q_source = q_source
        self._q_target = q_target
        self._h_source = h_source
        self._h_target = h_target

    @property
    def description(self) -> str:
        return f"nordsieck_rescaled_accuracy/{self._label}"

    def check(self) -> None:
        inst = self._instance
        h = self._h_target
        t_end = 1.0

        z_terms = [Tensor([1.0, 0.0])]
        for j in range(1, self._q_source + 1):
            deriv = (-_BASE_K) ** j
            scale = (self._h_source**j) / math.factorial(j)
            z_terms.append(Tensor([scale * deriv, -scale * deriv]))
        z_source = tuple(z_terms)
        transformed = (
            NordsieckState(0.0, self._h_source, z_source)
            .change_order(self._q_target)
            .rescale_step(h)
        )

        state = transformed
        for _ in range(round((t_end - state.t) / h)):
            state = inst.step(self._rhs, state, h)
        transformed_error = _max_base_network_error(state.u, state.t)

        fresh = inst.init_state(self._rhs, 0.0, Tensor([1.0, 0.0]), h)
        for _ in range(round((t_end - fresh.t) / h)):
            fresh = inst.step(self._rhs, fresh, h)
        fresh_error = _max_base_network_error(fresh.u, fresh.t)

        assert transformed_error <= 2.0 * fresh_error, (
            f"{self._label}: transformed error {transformed_error:.3e} "
            f"> 2x fresh-init error {fresh_error:.3e}"
        )


_NORDSIECK_ROUND_TRIP_CLAIMS: list[_NordsieckRoundTripClaim] = [
    _NordsieckRoundTripClaim(),
]

_NORDSIECK_RESCALED_ACCURACY_CLAIMS: list[_NordsieckRescaledAccuracyClaim] = [
    _NordsieckRescaledAccuracyClaim(
        bdf2,
        "bdf2",
        _BASE_RHS,
    ),
    _NordsieckRescaledAccuracyClaim(
        adams_moulton2,
        "am2",
        BlackBoxRHS(_base_network_f),
    ),
]


@pytest.mark.parametrize(
    "claim",
    _NORDSIECK_ROUND_TRIP_CLAIMS,
    ids=[c.description for c in _NORDSIECK_ROUND_TRIP_CLAIMS],
)
def test_nordsieck_round_trip(claim: _NordsieckRoundTripClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _NORDSIECK_RESCALED_ACCURACY_CLAIMS,
    ids=[c.description for c in _NORDSIECK_RESCALED_ACCURACY_CLAIMS],
)
def test_nordsieck_rescaled_accuracy(
    claim: _NordsieckRescaledAccuracyClaim,
) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 10 claims — variable-order Nordsieck integrators
# ---------------------------------------------------------------------------


def _sharpening_decay_f(t: float, u: Tensor) -> Tensor:
    k2 = 1.0 if t < 0.5 else 10.0
    x0, x1 = float(u[0]), float(u[1])
    return Tensor([-x0, x0 - k2 * x1, k2 * x1], backend=u.backend)


class _VariableOrderClimbClaim(Claim):
    """Verify variable-order Adams climbs on a smooth non-stiff network."""

    @property
    def description(self) -> str:
        return "variable_order/climbs_on_smooth_network"

    def check(self) -> None:
        selector = OrderSelector(q_min=2, q_max=4, atol=1e-4, rtol=1e-4)
        inst = VariableOrderNordsieckIntegrator(adams_family, selector)
        state = inst.advance(
            BlackBoxRHS(_decay_f),
            Tensor([1.0, 0.0, 0.0]),
            t0=0.0,
            t_end=1.5,
            dt0=0.025,
        )

        assert max(inst.accepted_orders) == 4
        assert inst.accepted_orders[-1] == 4

        x0_ex, x1_ex, x2_ex = _decay_exact(state.t)
        err = max(
            abs(float(state.u[0]) - x0_ex),
            abs(float(state.u[1]) - x1_ex),
            abs(float(state.u[2]) - x2_ex),
        )
        assert err < 5e-4

        total = sum(float(state.u[i]) for i in range(3))
        assert abs(total - 1.0) < 1e-12


class _VariableOrderDropClaim(Claim):
    """Verify variable-order Adams lowers order when the network sharpens."""

    @property
    def description(self) -> str:
        return "variable_order/drops_on_sharpening_network"

    def check(self) -> None:
        selector = OrderSelector(q_min=2, q_max=4, atol=1e-4, rtol=1e-4)
        inst = VariableOrderNordsieckIntegrator(
            adams_family,
            selector,
            q_initial=4,
        )
        state = inst.advance(
            BlackBoxRHS(_sharpening_decay_f),
            Tensor([1.0, 0.0, 0.0]),
            t0=0.0,
            t_end=1.0,
            dt0=0.02,
        )

        post_sharpen_orders = [
            q
            for t, q in zip(inst.accepted_times, inst.accepted_orders, strict=True)
            if t > 0.55
        ]
        assert post_sharpen_orders
        assert min(post_sharpen_orders) == 2

        total = sum(float(state.u[i]) for i in range(3))
        assert abs(total - 1.0) < 1e-12
        for i in range(3):
            assert float(state.u[i]) >= -1e-10


_VARIABLE_ORDER_CLAIMS: list[Claim] = [
    _VariableOrderClimbClaim(),
    _VariableOrderDropClaim(),
]


@pytest.mark.parametrize(
    "claim",
    _VARIABLE_ORDER_CLAIMS,
    ids=[c.description for c in _VARIABLE_ORDER_CLAIMS],
)
def test_variable_order_nordsieck(claim: Claim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 11 claims — stiffness detection and family switching
# ---------------------------------------------------------------------------


def _stiffening_decay_f(t: float, u: Tensor) -> Tensor:
    k2 = 1.0 if t < 0.5 else 80.0
    x0, x1 = float(u[0]), float(u[1])
    return Tensor([-x0, x0 - k2 * x1, k2 * x1], backend=u.backend)


def _stiffening_decay_jac(t: float, u: Tensor) -> Tensor:
    k2 = 1.0 if t < 0.5 else 80.0
    return Tensor([[-1.0, 0.0, 0.0], [1.0, -k2, 0.0], [0.0, k2, 0.0]])


_STIFFENING_RHS = JacobianRHS(f=_stiffening_decay_f, jac=_stiffening_decay_jac)


class _StiffnessDiagnosticClaim(Claim):
    """Verify the Gershgorin stiffness estimate crosses the expected threshold."""

    @property
    def description(self) -> str:
        return "stiffness/diagnostic_threshold"

    def check(self) -> None:
        diagnostic = StiffnessDiagnostic()
        jac = Tensor([[-3.0, 1.0], [0.5, -2.0]])

        below = diagnostic.update(jac, h=0.1)
        above = diagnostic.update(jac, h=0.3)

        assert abs(below - 0.4) < 1e-15
        assert abs(above - 1.2) < 1e-15
        assert below < 1.0 < above


class _FamilySwitchRoundTripClaim(Claim):
    """Verify Adams → BDF → Adams preserves the represented solution."""

    @property
    def description(self) -> str:
        return "stiffness/family_switch_round_trip"

    def check(self) -> None:
        inst = FamilySwitchingNordsieckIntegrator(
            adams_family=adams_family,
            bdf_family=bdf_family,
            switcher=StiffnessSwitcher(),
            q=2,
            initial_family="adams",
        )
        state = NordsieckState(
            0.25,
            0.05,
            (
                Tensor([0.8, 0.15, 0.05]),
                Tensor([-0.02, 0.01, 0.01]),
                Tensor([0.0, 0.0, 0.0]),
            ),
        )

        bdf_state = inst.transform_family(state, "bdf")
        round_trip = inst.transform_family(bdf_state, "adams")

        assert nordsieck_solution_distance(state, round_trip) == 0.0
        assert round_trip.h == state.h
        assert round_trip.q == state.q


class _StiffeningNetworkSwitchClaim(Claim):
    """Verify the switcher chooses BDF after a synthetic network stiffens."""

    @property
    def description(self) -> str:
        return "stiffness/switcher_fires_on_stiffening_network"

    def check(self) -> None:
        inst = FamilySwitchingNordsieckIntegrator(
            adams_family=adams_family,
            bdf_family=bdf_family,
            switcher=StiffnessSwitcher(stiff_threshold=1.0, nonstiff_threshold=0.4),
            q=2,
            initial_family="adams",
        )
        state = inst.advance(
            _STIFFENING_RHS,
            Tensor([1.0, 0.0, 0.0]),
            t0=0.0,
            t_end=0.8,
            dt=0.02,
        )

        pre_families = [
            family
            for t, family in zip(
                inst.accepted_times, inst.accepted_families, strict=True
            )
            if t < 0.45
        ]
        post_families = [
            family
            for t, family in zip(
                inst.accepted_times, inst.accepted_families, strict=True
            )
            if t > 0.55
        ]

        assert pre_families
        assert post_families
        assert set(pre_families) == {"adams"}
        assert "bdf" in post_families
        assert inst.switch_count >= 1
        assert max(inst.accepted_stiffness) > 1.0

        total = sum(float(state.u[i]) for i in range(3))
        assert abs(total - 1.0) < 1e-12
        for i in range(3):
            assert float(state.u[i]) >= -1e-10


_STIFFNESS_CLAIMS: list[Claim] = [
    _StiffnessDiagnosticClaim(),
    _FamilySwitchRoundTripClaim(),
    _StiffeningNetworkSwitchClaim(),
]


@pytest.mark.parametrize(
    "claim",
    _STIFFNESS_CLAIMS,
    ids=[c.description for c in _STIFFNESS_CLAIMS],
)
def test_stiffness_switching(claim: Claim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 12 claims — VODE-style adaptive controller
# ---------------------------------------------------------------------------


def _vode_fast_slow_f(t: float, u: Tensor) -> Tensor:
    k2 = 1.0 if t < 0.45 else 1000.0
    x0, x1 = float(u[0]), float(u[1])
    return Tensor([-x0, x0 - k2 * x1, k2 * x1], backend=u.backend)


def _vode_fast_slow_jac(t: float, u: Tensor) -> Tensor:
    k2 = 1.0 if t < 0.45 else 1000.0
    return Tensor([[-1.0, 0.0, 0.0], [1.0, -k2, 0.0], [0.0, k2, 0.0]])


_VODE_RHS = JacobianRHS(f=_vode_fast_slow_f, jac=_vode_fast_slow_jac)


class _VODEControllerSwitchClaim(Claim):
    """Verify VODE policy composes order selection with stiffness switching."""

    @property
    def description(self) -> str:
        return "vode_controller/fast_slow_family_policy"

    def check(self) -> None:
        controller = VODEController(
            adams_family=adams_family,
            bdf_family=bdf_family,
            order_selector=OrderSelector(
                q_min=2,
                q_max=4,
                atol=5e-4,
                rtol=5e-4,
                factor_min=0.25,
                factor_max=1.15,
            ),
            stiffness_switcher=StiffnessSwitcher(
                stiff_threshold=1.0,
                nonstiff_threshold=0.4,
            ),
            q_initial=2,
            initial_family="adams",
        )
        state = controller.advance(
            _VODE_RHS,
            Tensor([1.0, 0.0, 0.0]),
            t0=0.0,
            t_end=0.7,
            dt0=0.005,
        )

        early_families = [
            family
            for t, family in zip(
                controller.accepted_times,
                controller.accepted_families,
                strict=True,
            )
            if t < 0.35
        ]
        late_families = [
            family
            for t, family in zip(
                controller.accepted_times,
                controller.accepted_families,
                strict=True,
            )
            if t > 0.5
        ]

        assert early_families
        assert late_families
        assert set(early_families) == {"adams"}
        assert "bdf" in late_families
        assert controller.family_switches >= 1
        assert max(controller.accepted_stiffness) > 1.0
        assert max(controller.accepted_orders) > 2

        total = sum(float(state.u[i]) for i in range(3))
        assert abs(total - 1.0) < 1e-12
        for i in range(3):
            assert float(state.u[i]) >= -1e-10


_VODE_CONTROLLER_CLAIMS: list[Claim] = [
    _VODEControllerSwitchClaim(),
]


@pytest.mark.parametrize(
    "claim",
    _VODE_CONTROLLER_CLAIMS,
    ids=[c.description for c in _VODE_CONTROLLER_CLAIMS],
)
def test_vode_controller(claim: Claim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 13 claims — exponential integrators
# ---------------------------------------------------------------------------


def _etd_split_rhs() -> LinearPlusNonlinearRHS:
    linear = Tensor([[-8.0, 0.0, 0.0], [8.0, -0.5, 0.0], [0.0, 0.5, 0.0]])

    def residual(t: float, u: Tensor) -> Tensor:
        # Additional slow production path X0 -> X2.  This keeps the total
        # Jacobian column sums zero when combined with the diagonal-linear
        # capture chain above, so mass fractions remain conserved.
        x0 = float(u[0])
        rate = 0.25 + 0.1 * math.sin(2.0 * t)
        return Tensor([-rate * x0, 0.0, rate * x0], backend=u.backend)

    return LinearPlusNonlinearRHS(linear, residual)


def _integrate_etd(
    inst: (
        ExponentialEulerIntegrator
        | ETDRK2Integrator
        | CoxMatthewsETDRK4Integrator
        | KrogstadETDRK4Integrator
    ),
    dt: float,
    t_end: float = 0.5,
) -> RKState:
    rhs = _etd_split_rhs()
    state = RKState(0.0, Tensor([1.0, 0.0, 0.0]))
    n_steps = round(t_end / dt)
    for _ in range(n_steps):
        state = inst.step(rhs, state, dt)
    return state


class _PhiFunctionClaim(Claim):
    """Verify phi-function coefficient algebra on a nilpotent matrix."""

    @property
    def description(self) -> str:
        return "exponential/phi_function_coefficients"

    def check(self) -> None:
        A = Tensor([[0.0, 1.0], [0.0, 0.0]])
        v = Tensor([0.0, 1.0])

        phi0 = PhiFunction(0).apply(A, v)
        phi1 = PhiFunction(1).apply(A, v)
        phi2 = PhiFunction(2).apply(A, v)
        phi3 = PhiFunction(3).apply(A, v)

        assert float(norm(phi0 - Tensor([1.0, 1.0]))) < 1e-14
        assert float(norm(phi1 - Tensor([0.5, 1.0]))) < 1e-14
        assert float(norm(phi2 - Tensor([1.0 / 6.0, 0.5]))) < 1e-14
        assert float(norm(phi3 - Tensor([1.0 / 24.0, 1.0 / 6.0]))) < 1e-14


class _ETDConvergenceClaim(Claim):
    """Verify ETD convergence on a mass-conserving synthetic abundance split."""

    def __init__(
        self,
        instance: (
            ExponentialEulerIntegrator
            | ETDRK2Integrator
            | CoxMatthewsETDRK4Integrator
            | KrogstadETDRK4Integrator
        ),
        label: str,
        order: int,
    ) -> None:
        self._instance = instance
        self._label = label
        self._order = order

    @property
    def description(self) -> str:
        return f"exponential/convergence/{self._label}"

    def check(self) -> None:
        dts = [0.025, 0.0125, 0.00625]
        reference = _integrate_etd(self._instance, dt=0.0015625)
        errors = [
            float(norm(_integrate_etd(self._instance, dt).u - reference.u))
            for dt in dts
        ]
        log_dts = [math.log(dt) for dt in dts]
        log_errs = [math.log(err) for err in errors]
        mean_x = sum(log_dts) / len(log_dts)
        mean_y = sum(log_errs) / len(log_errs)
        slope = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(log_dts, log_errs, strict=True)
        ) / sum((x - mean_x) ** 2 for x in log_dts)

        assert slope >= self._order - 0.25, (
            f"{self._label}: convergence slope {slope:.3f} < "
            f"declared order {self._order}"
        )

        final = _integrate_etd(self._instance, dts[-1])
        _assert_mass_fractions(final.u, label=self._label, n=3, conservation_tol=1e-10)


_EXPONENTIAL_CLAIMS: list[Claim] = [
    _PhiFunctionClaim(),
    _ETDConvergenceClaim(etd_euler, "etd_euler", order=1),
    _ETDConvergenceClaim(etdrk2, "etdrk2", order=2),
    _ETDConvergenceClaim(cox_matthews_etdrk4, "cox_matthews_etdrk4", order=4),
    _ETDConvergenceClaim(krogstad_etdrk4, "krogstad_etdrk4", order=4),
]


@pytest.mark.parametrize(
    "claim",
    _EXPONENTIAL_CLAIMS,
    ids=[c.description for c in _EXPONENTIAL_CLAIMS],
)
def test_exponential_integrators(claim: Claim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 14 claims — operator splitting
# ---------------------------------------------------------------------------

# Test problem: 2D split oscillator.
#
# Full system: du/dt = (A + B) u  with  A = [[0, -w], [0, 0]],
#                                        B = [[0,  0], [w, 0]],  w = 1.
#
# [A, B] = AB - BA = [[w², 0], [0, 0]] ≠ 0, so splitting introduces a
# nonzero commutator error that reveals the order of the splitting scheme.
#
# Exact solution of the full system: rotation by angle w*t, i.e.
#   u(t) = [[cos(wt), -sin(wt)], [sin(wt), cos(wt)]] @ u0.
#
# Sub-integrators are rk4 (order 4) so sub-integrator error is O(h⁴),
# well below the splitting error at the test step sizes.

_SPLIT_OMEGA = 1.0


def _split_component_a(t: float, u: Tensor) -> Tensor:
    """Component A: du1/dt = -w*u2, du2/dt = 0."""
    return Tensor([-_SPLIT_OMEGA * float(u[1]), 0.0], backend=u.backend)


def _split_component_b(t: float, u: Tensor) -> Tensor:
    """Component B: du1/dt = 0, du2/dt = +w*u1."""
    return Tensor([0.0, _SPLIT_OMEGA * float(u[0])], backend=u.backend)


def _split_rhs() -> OperatorSplitRHS:
    from cosmic_foundry.computation.time_integrators import BlackBoxRHS

    return OperatorSplitRHS(
        [BlackBoxRHS(_split_component_a), BlackBoxRHS(_split_component_b)]
    )


def _split_exact(t: float, u0: Tensor) -> Tensor:
    """Exact solution of the full oscillator at time t."""
    c, s = math.cos(_SPLIT_OMEGA * t), math.sin(_SPLIT_OMEGA * t)
    u0_0, u0_1 = float(u0[0]), float(u0[1])
    return Tensor([c * u0_0 - s * u0_1, s * u0_0 + c * u0_1], backend=u0.backend)


def _integrate_split(
    integrator: StrangSplittingIntegrator,
    dt: float,
    t_end: float = 1.0,
) -> RKState:
    rhs = _split_rhs()
    u0 = Tensor([1.0, 0.0])
    state = RKState(0.0, u0)
    n_steps = round(t_end / dt)
    for _ in range(n_steps):
        state = integrator.step(rhs, state, dt)
    return state


class _SplittingConvergenceClaim(Claim):
    """Verify splitting convergence order on the 2D split oscillator."""

    def __init__(
        self,
        sequence: list[SplittingStep],
        label: str,
        order: int,
    ) -> None:
        self._sequence = sequence
        self._label = label
        self._order = order

    @property
    def description(self) -> str:
        return f"splitting/convergence/{self._label}"

    def check(self) -> None:
        integrator = StrangSplittingIntegrator(
            [rk4, rk4], self._sequence, order=self._order
        )
        u0 = Tensor([1.0, 0.0])
        t_end = 1.0
        dts = [0.1, 0.05, 0.025]
        errors = [
            float(
                norm(
                    _integrate_split(integrator, dt, t_end).u - _split_exact(t_end, u0)
                )
            )
            for dt in dts
        ]
        log_dts = [math.log(dt) for dt in dts]
        log_errs = [math.log(e) for e in errors]
        mean_x = sum(log_dts) / len(log_dts)
        mean_y = sum(log_errs) / len(log_errs)
        slope = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(log_dts, log_errs, strict=True)
        ) / sum((x - mean_x) ** 2 for x in log_dts)
        assert slope >= self._order - 0.25, (
            f"{self._label}: convergence slope {slope:.3f} < "
            f"declared order {self._order}"
        )


_SPLITTING_CLAIMS: list[Claim] = [
    _SplittingConvergenceClaim(lie_steps(), "lie", order=1),
    _SplittingConvergenceClaim(strang_steps(), "strang", order=2),
]


@pytest.mark.parametrize(
    "claim",
    _SPLITTING_CLAIMS,
    ids=[c.description for c in _SPLITTING_CLAIMS],
)
def test_splitting_integrators(claim: Claim) -> None:
    claim.check()
