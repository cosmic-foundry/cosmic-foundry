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

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators import (
    BlackBoxRHS,
    ConstantStep,
    DIRKIntegrator,
    HamiltonianSplit,
    JacobianRHS,
    PartitionedState,
    PIController,
    RKState,
    RungeKuttaIntegrator,
    SymplecticSplittingIntegrator,
    TimeStepper,
    backward_euler,
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
