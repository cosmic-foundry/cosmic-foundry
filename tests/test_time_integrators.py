"""Verification for the time-integration layer.

Claim classes and auto-discovery framework for TimeIntegrator subclasses:

  _RKOrderClaim          — B-series order conditions via rooted-tree enumeration
  _ConvergenceClaim      — temporal convergence rate on dy/dt = λy
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
    DIRKIntegrator,
    FamilySwitchingNordsieckIntegrator,
    HamiltonianSplit,
    IMEXIntegrator,
    JacobianRHS,
    NordsieckIntegrator,
    NordsieckState,
    OrderSelector,
    PartitionedState,
    PIController,
    RKState,
    RungeKuttaIntegrator,
    StiffnessDiagnostic,
    StiffnessSwitcher,
    SymplecticSplittingIntegrator,
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
    crouzeix_3,
    dormand_prince,
    elementary_weight,
    forest_ruth,
    forward_euler,
    gamma,
    heun,
    implicit_midpoint,
    leapfrog,
    midpoint,
    nordsieck_solution_distance,
    ralston,
    rk4,
    stability_function,
    symplectic_euler,
    trees_up_to_order,
    yoshida_6,
    yoshida_8,
)
from tests.claims import Claim

# ---------------------------------------------------------------------------
# Claim classes
# ---------------------------------------------------------------------------

_DT_BASE = 0.1
_N_HALVINGS = 5


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
            state = RKState(0.0, Tensor([1.0]))
            for _ in range(n_steps):
                state = inst.step(rhs, state, dt)
            exact = math.exp(lam * state.t)
            errors.append(abs(float(state.u[0]) - exact))

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

        final = stepper.advance(rhs, Tensor([1.0]), 0.0, self._t_end)

        assert (
            abs(final.t - self._t_end) < 1e-12
        ), f"{self._label}: final time {final.t} != t_end {self._t_end}"
        exact = math.exp(lam * self._t_end)
        rel_err = abs(float(final.u[0]) - exact) / abs(exact)
        assert (
            rel_err < self._rtol
        ), f"{self._label}: relative error {rel_err:.2e} > rtol {self._rtol}"


# ---------------------------------------------------------------------------
# Phase 1 claims — PIController
# ---------------------------------------------------------------------------

# Default PI exponents (Hairer Vol. II recommendation): α = 0.7/p, β = 0.4/p.
# These give a well-damped response for explicit RK methods on smooth problems.
_PI_ALPHA = 0.7
_PI_BETA = 0.4


class _PIAccuracyClaim(Claim):
    """Verify that PIController achieves error ≤ c_rel · tol on dy/dt = λy.

    Integrates dy/dt = λy from 0 to t_end with a PIController and checks
    that the global error satisfies |y_h(t_end) - exp(λ t_end)| ≤ c_rel · tol.
    The c_rel factor absorbs pre-asymptotic constants; 10 is generous but
    distinguishes correct control from uncontrolled growth.
    """

    def __init__(
        self,
        instance: RungeKuttaIntegrator,
        label: str,
        tol: float = 1e-4,
        lam: float = -1.0,
        t_end: float = 1.0,
        c_rel: float = 10.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._tol = tol
        self._lam = lam
        self._t_end = t_end
        self._c_rel = c_rel

    @property
    def description(self) -> str:
        return f"pi_accuracy/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam = self._lam
        p = inst.order
        rhs = BlackBoxRHS(lambda t, u, _lam=lam: u * _lam)
        pi = PIController(
            alpha=_PI_ALPHA / p,
            beta=_PI_BETA / p,
            tol=self._tol,
            dt0=0.1,
        )
        stepper = TimeStepper(inst, controller=pi)
        final = stepper.advance(rhs, Tensor([1.0]), 0.0, self._t_end)
        exact = math.exp(lam * self._t_end)
        err = abs(float(final.u[0]) - exact)
        assert (
            err <= self._c_rel * self._tol
        ), f"{self._label}: error {err:.2e} > {self._c_rel} × tol {self._tol:.2e}"


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
        lam: float = -1.0,
        t_end: float = 1.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._tol_coarse = tol_coarse
        self._tol_fine = tol_fine
        self._lam = lam
        self._t_end = t_end

    @property
    def description(self) -> str:
        return f"pi_work_precision/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam = self._lam
        p = inst.order
        rhs = BlackBoxRHS(lambda t, u, _lam=lam: u * _lam)
        exact = math.exp(lam * self._t_end)

        def _run(tol: float) -> float:
            pi = PIController(
                alpha=_PI_ALPHA / p,
                beta=_PI_BETA / p,
                tol=tol,
                dt0=0.1,
            )
            final = TimeStepper(inst, controller=pi).advance(
                rhs, Tensor([1.0]), 0.0, self._t_end
            )
            return abs(float(final.u[0]) - exact)

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
# Phase 3 claims — symplectic splitting
# ---------------------------------------------------------------------------

# Harmonic oscillator: H = p²/2 + q²/2, exact solution q(t)=cos t, p(t)=−sin t
# starting from q(0)=1, p(0)=0.
_HO = HamiltonianSplit(dT_dp=lambda p: p, dV_dq=lambda q: q)

# Larger starting dt than _DT_BASE: high-order methods reach machine precision
# too quickly with dt=0.1, so the regression would include noise-floor points.
_DT_BASE_SYM = 0.5
_N_HALVINGS_SYM = 4


class _SplittingOrderClaim(Claim):
    """Verify convergence order on the unit harmonic oscillator at t=1."""

    def __init__(self, instance: SymplecticSplittingIntegrator, label: str) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"symplectic_order/{self._label}"

    def check(self) -> None:
        inst = self._instance
        dts = [_DT_BASE_SYM / (2**k) for k in range(_N_HALVINGS_SYM + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt)
            state = PartitionedState(0.0, Tensor([1.0]), Tensor([0.0]))
            for _ in range(n_steps):
                state = inst.step(_HO, state, dt)
            # Exact: q(t)=cos t, p(t)=−sin t
            q_exact = math.cos(state.t)
            p_exact = -math.sin(state.t)
            eq = abs(float(state.q[0]) - q_exact)
            ep = abs(float(state.p[0]) - p_exact)
            errors.append(max(eq, ep))

        eps = sys.float_info.epsilon * 100
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

        assert slope >= inst.order - 0.4, (
            f"{self._label}: convergence slope {slope:.3f} < declared order "
            f"{inst.order} - 0.4"
        )


class _EnergyBoundClaim(Claim):
    """Verify that the Hamiltonian error is bounded over a long integration.

    Symplectic integrators conserve a modified Hamiltonian, so the energy
    error H(qₙ, pₙ) − H(q₀, p₀) stays O(dtᵖ) rather than growing linearly
    in time.  The claim integrates for n_periods full periods at a fixed step
    size and checks that the maximum relative energy deviation is < tol.
    """

    def __init__(
        self,
        instance: SymplecticSplittingIntegrator,
        label: str,
        dt: float = 0.1,
        n_periods: int = 20,
        tol: float = 0.05,
    ) -> None:
        self._instance = instance
        self._label = label
        self._dt = dt
        self._n_periods = n_periods
        self._tol = tol

    @property
    def description(self) -> str:
        return f"energy_bound/{self._label}"

    def check(self) -> None:
        inst = self._instance
        dt = self._dt
        state = PartitionedState(0.0, Tensor([1.0]), Tensor([0.0]))
        H0 = 0.5 * float(state.q[0]) ** 2 + 0.5 * float(state.p[0]) ** 2  # = 0.5
        n_steps = round(self._n_periods * 2 * math.pi / dt)
        max_err = 0.0
        for _ in range(n_steps):
            state = inst.step(_HO, state, dt)
            H = 0.5 * float(state.q[0]) ** 2 + 0.5 * float(state.p[0]) ** 2
            max_err = max(max_err, abs(H - H0) / H0)
        assert (
            max_err < self._tol
        ), f"{self._label}: max relative energy error {max_err:.3e} >= tol {self._tol}"


_SPLITTING_ORDER_CLAIMS: list[_SplittingOrderClaim] = [
    _SplittingOrderClaim(symplectic_euler, "symplectic_euler"),
    _SplittingOrderClaim(leapfrog, "leapfrog"),
    _SplittingOrderClaim(forest_ruth, "forest_ruth"),
    _SplittingOrderClaim(yoshida_6, "yoshida_6"),
    _SplittingOrderClaim(yoshida_8, "yoshida_8"),
]

_ENERGY_BOUND_CLAIMS: list[_EnergyBoundClaim] = [
    _EnergyBoundClaim(leapfrog, "leapfrog", dt=0.1, tol=0.005),
    _EnergyBoundClaim(forest_ruth, "forest_ruth", dt=0.3, tol=0.01),
    _EnergyBoundClaim(yoshida_6, "yoshida_6", dt=0.5, tol=0.01),
    _EnergyBoundClaim(yoshida_8, "yoshida_8", dt=0.5, tol=0.01),
]


@pytest.mark.parametrize(
    "claim",
    _SPLITTING_ORDER_CLAIMS,
    ids=[c.description for c in _SPLITTING_ORDER_CLAIMS],
)
def test_symplectic_order(claim: _SplittingOrderClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _ENERGY_BOUND_CLAIMS,
    ids=[c.description for c in _ENERGY_BOUND_CLAIMS],
)
def test_energy_bound(claim: _EnergyBoundClaim) -> None:
    claim.check()


# ---------------------------------------------------------------------------
# Phase 4 claims — DIRK integrators
# ---------------------------------------------------------------------------


class _DIRKConvergenceClaim(Claim):
    """Verify convergence order on dy/dt = λy using JacobianRHS."""

    def __init__(
        self,
        instance: DIRKIntegrator,
        label: str,
        lam: float = -1.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._lam = lam

    @property
    def description(self) -> str:
        return f"dirk_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam = self._lam
        rhs = JacobianRHS(
            f=lambda t, u, _lam=lam: _lam * u,
            jac=lambda t, u, _lam=lam: Tensor([[_lam]]),
        )
        dts = [_DT_BASE / (2**k) for k in range(_N_HALVINGS + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt)
            state = RKState(0.0, Tensor([1.0]))
            for _ in range(n_steps):
                state = inst.step(rhs, state, dt)
            exact = math.exp(lam * state.t)
            errors.append(abs(float(state.u[0]) - exact))

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
    """Verify IMEX convergence order on dy/dt = f_E(y) + f_I(y) = (λ_E + λ_I)y."""

    def __init__(
        self,
        instance: IMEXIntegrator,
        label: str,
        lam_E: float = -1.0,
        lam_I: float = -2.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._lam_E = lam_E
        self._lam_I = lam_I

    @property
    def description(self) -> str:
        return f"imex_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam_E, lam_I = self._lam_E, self._lam_I
        lam_tot = lam_E + lam_I
        rhs = AdditiveRHS(
            f_E=lambda t, u, _l=lam_E: _l * u,
            f_I=lambda t, u, _l=lam_I: _l * u,
            jac_I=lambda t, u, _l=lam_I: Tensor([[_l]]),
        )
        dts = [_DT_BASE / (2**k) for k in range(_N_HALVINGS + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt)
            state = RKState(0.0, Tensor([1.0]))
            for _ in range(n_steps):
                state = inst.step(rhs, state, dt)
            exact = math.exp(lam_tot * state.t)
            errors.append(abs(float(state.u[0]) - exact))

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
    """Verify Adams-Bashforth convergence rate on dy/dt = λy.

    The first k−1 steps are bootstrapped with RK4 internally; the slope
    is measured across the full dt grid and reflects the AB order because
    the RK4 bootstrap error is the same order as AB4 and higher for AB2/AB3.
    """

    def __init__(
        self,
        instance: AdamsBashforthIntegrator,
        label: str,
        lam: float = -1.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._lam = lam

    @property
    def description(self) -> str:
        return f"ab_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam = self._lam
        rhs = BlackBoxRHS(lambda t, u, _lam=lam: _lam * u)
        dts = [_DT_BASE / (2**k) for k in range(_N_HALVINGS + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt)
            state = ABState(0.0, Tensor([1.0]))
            for _ in range(n_steps):
                state = inst.step(rhs, state, dt)
            exact = math.exp(lam * state.t)
            errors.append(abs(float(state.u[0]) - exact))

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
    """Verify BDF-q convergence rate on dy/dt = λy.

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
        lam: float = -1.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._lam = lam

    @property
    def description(self) -> str:
        return f"bdf_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam = self._lam
        rhs = JacobianRHS(
            f=lambda t, u, _l=lam: _l * u,
            jac=lambda t, u, _l=lam: Tensor([[_l]], backend=u.backend),
        )
        dt_base = _DT_BASE / inst.order
        n_halvings = _BDF_N_HALVINGS - (math.floor(math.log2(inst.order)) + 1)
        dts = [dt_base / (2**k) for k in range(n_halvings + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt) - inst.order
            state = inst.init_state(rhs, 0.0, Tensor([1.0]), dt)
            for _ in range(max(n_steps, 1)):
                state = inst.step(rhs, state, dt)
            exact = math.exp(lam * state.t)
            errors.append(abs(float(state.u[0]) - exact))

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
    """Verify Adams-Moulton-q convergence rate on dy/dt = λy.

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
        lam: float = -1.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._lam = lam

    @property
    def description(self) -> str:
        return f"am_convergence/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam = self._lam
        rhs = BlackBoxRHS(lambda t, u, _l=lam: _l * u)
        dt_base = _DT_BASE / inst.order
        n_halvings = _AM_N_HALVINGS - (math.floor(math.log2(inst.order)) + 1)
        dts = [dt_base / (2**k) for k in range(n_halvings + 1)]
        errors: list[float] = []
        for dt in dts:
            n_steps = math.ceil(1.0 / dt) - inst.order
            state = inst.init_state(rhs, 0.0, Tensor([1.0]), dt)
            for _ in range(max(n_steps, 1)):
                state = inst.step(rhs, state, dt)
            exact = math.exp(lam * state.t)
            errors.append(abs(float(state.u[0]) - exact))

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
            z=(Tensor([1.0]), Tensor([-0.1]), Tensor([0.005])),
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
        lam: float = -1.0,
    ) -> None:
        self._instance = instance
        self._label = label
        self._rhs = rhs
        self._q_source = q_source
        self._q_target = q_target
        self._h_source = h_source
        self._h_target = h_target
        self._lam = lam

    @property
    def description(self) -> str:
        return f"nordsieck_rescaled_accuracy/{self._label}"

    def check(self) -> None:
        inst = self._instance
        lam = self._lam
        h = self._h_target
        t_end = 1.0

        z_source = tuple(
            Tensor([(self._h_source**j) * (lam**j) / math.factorial(j)])
            for j in range(self._q_source + 1)
        )
        transformed = (
            NordsieckState(0.0, self._h_source, z_source)
            .change_order(self._q_target)
            .rescale_step(h)
        )

        state = transformed
        for _ in range(round((t_end - state.t) / h)):
            state = inst.step(self._rhs, state, h)
        transformed_error = abs(float(state.u[0]) - math.exp(lam * state.t))

        fresh = inst.init_state(self._rhs, 0.0, Tensor([1.0]), h)
        for _ in range(round((t_end - fresh.t) / h)):
            fresh = inst.step(self._rhs, fresh, h)
        fresh_error = abs(float(fresh.u[0]) - math.exp(lam * fresh.t))

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
        JacobianRHS(
            f=lambda t, u: -1.0 * u,
            jac=lambda t, u: Tensor([[-1.0]], backend=u.backend),
        ),
    ),
    _NordsieckRescaledAccuracyClaim(
        adams_moulton2,
        "am2",
        BlackBoxRHS(lambda t, u: -1.0 * u),
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
