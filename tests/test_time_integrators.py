"""Verification for the time-integration layer.

Claim classes and parametric test framework for TimeIntegrator subclasses.
Problem setups, exact solutions, and convergence helpers live in
tests/integrator_fixtures.py; network construction helpers live in
tests/parametric_networks.py.

Claim families and their test functions:

  _RKOrderClaim          — B-series order conditions via rooted-tree enumeration
  _ConvergenceClaim      — convergence rate (all families: RK, DIRK, IMEX, AB,
                           BDF, AM, ETD, splitting, symplectic, reaction-network)
  _ConservationClaim     — abundance conservation (RK, IMEX, AB, BDF, AM)
  _StepperClaim          — end-to-end TimeStepper.advance accuracy
  _PIAccuracyClaim       — PIController achieves error ≤ c_rel · tol
  _PIWorkPrecisionClaim  — tighter tol yields smaller error
  _AStabilityClaim       — A-stability on the imaginary axis
  _LStabilityClaim       — L-stability at Re(z) → −∞
  [Nordsieck, variable-order, stiffness, VODE, ETD phi-fn, AutoDispatch,
   type coherence, Newton, network invariants, project_conserved,
   projected Newton, constraint lifecycle, NSE solver, rate-threshold guard,
   parametric network, cost model, offline network]
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation.backends import (
    NumpyBackend,
    get_default_backend,
    set_default_backend,
)
from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators import (
    AdditiveRungeKuttaIntegrator,
    AutoIntegrator,
    BlackBoxRHS,
    CompositionIntegrator,
    ConstantStep,
    ConstraintAwareController,
    ExplicitMultistepIntegrator,
    FamilySwitchingNordsieckIntegrator,
    HamiltonianRHS,
    ImplicitRungeKuttaIntegrator,
    JacobianRHS,
    MultistepIntegrator,
    NordsieckHistory,
    ODEState,
    OrderSelector,
    PhiFunction,
    PIController,
    ReactionNetworkRHS,
    RungeKuttaIntegrator,
    StiffnessDiagnostic,
    StiffnessSwitcher,
    SymplecticCompositionIntegrator,
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
    forest_ruth,
    forward_euler,
    gamma,
    heun,
    implicit_midpoint,
    krogstad_etdrk4,
    leapfrog,
    lie_steps,
    midpoint,
    nonlinear_solve,
    nordsieck_solution_distance,
    project_conserved,
    ralston,
    rk4,
    solve_nse,
    stability_function,
    strang_steps,
    symplectic_euler,
    trees_up_to_order,
    yoshida_6,
    yoshida_8,
    yoshida_steps,
)
from cosmic_foundry.computation.time_integrators._newton import newton_solve
from cosmic_foundry.computation.time_integrators.integrator import TimeIntegrator
from tests.claims import Claim
from tests.integrator_fixtures import (
    BASE_RHS,
    BASE_RHS_IMEX,
    BDF_N_HALVINGS,
    DECAY_RHS,
    DECAY_RHS_IMEX,
    DT_BASE,
    N_HALVINGS,
    PI_ALPHA,
    PI_BETA,
    STIFFENING_RHS,
    VODE_RHS,
    assert_conservation,
    base_network_f,
    decay_exact,
    decay_f,
    etd_split_rhs,
    integrate_etd,
    integrate_split,
    max_base_network_error,
    max_decay_error,
    measure_convergence_slope,
    run_conservation,
    run_multistep,
    run_multistep_conservation,
    run_rk,
    sharpening_decay_f,
    split_exact,
    split_rhs,
)
from tests.parametric_networks import (
    _CostCalibration,
    _CostModelClaim,
    _ParametricNSEClaim,
    calibrate_cost,
    chain_claims,
    spoke_claims,
)

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

_PREV_BACKEND = get_default_backend()
set_default_backend(NumpyBackend())


@pytest.fixture(scope="module", autouse=True)
def _numpy_backend() -> Any:
    """Restore the original default backend after this module finishes."""
    yield
    set_default_backend(_PREV_BACKEND)


# ---------------------------------------------------------------------------
# Range knobs — CI and offline
# ---------------------------------------------------------------------------

_CHAIN_N = range(3, 5)  # n_species = 3, 4
_SPOKE_N = range(3, 7)  # n_species = 3, 4, 5, 6
_SPOKE_K = [1, 10]  # k_fast/k_slow ratios

_OFFLINE_NETWORK_STRESS = os.environ.get("COSMIC_FOUNDRY_OFFLINE_NETWORK_STRESS") == "1"
_CHAIN_N_OFFLINE = range(5, 12)  # n_species = 5 .. 11
_SPOKE_N_OFFLINE = range(7, 22)  # n_species = 7 .. 21
_SPOKE_K_OFFLINE = [1, 10, 100]  # k_fast/k_slow ratios


# ---------------------------------------------------------------------------
# Unified convergence claim
# ---------------------------------------------------------------------------


class _ConvergenceClaim(Claim):
    """Verify convergence order: log-log slope ≥ declared order − tolerance.

    run_fn(dt) must return a scalar error at t ≈ 1 (or a comparable fixed end
    point).  The slope is measured across the dt grid; machine-precision points
    are excluded.
    """

    def __init__(
        self,
        label: str,
        run_fn: Callable[[float], float],
        order: int,
        dt_base: float = DT_BASE,
        n_halvings: int = N_HALVINGS,
        tolerance: float = 0.1,
    ) -> None:
        self._label = label
        self._run_fn = run_fn
        self._order = order
        self._dt_base = dt_base
        self._n_halvings = n_halvings
        self._tolerance = tolerance

    @property
    def description(self) -> str:
        return f"convergence/{self._label}"

    def check(self) -> None:
        dts = [self._dt_base / (2**k) for k in range(self._n_halvings + 1)]
        errors = [self._run_fn(dt) for dt in dts]
        slope = measure_convergence_slope(errors, dts, self._label)
        assert slope >= self._order - self._tolerance, (
            f"{self._label}: convergence slope {slope:.3f} < "
            f"declared order {self._order} − {self._tolerance}"
        )


# ---------------------------------------------------------------------------
# Unified conservation claim
# ---------------------------------------------------------------------------


class _ConservationClaim(Claim):
    """Verify decay-chain accuracy, mass-fraction sum = 1, and positivity.

    run_fn(dt, t_end) must return an ODEState at approximately t_end.
    exact_fn(t) returns the n-tuple of exact species values at time t.
    """

    def __init__(
        self,
        label: str,
        run_fn: Callable[[float, float], ODEState],
        n: int,
        exact_fn: Callable[[float], tuple[float, ...]],
        dt: float = 0.05,
        t_end: float = 2.0,
        acc_tol: float = 1e-3,
    ) -> None:
        self._label = label
        self._run_fn = run_fn
        self._n = n
        self._exact_fn = exact_fn
        self._dt = dt
        self._t_end = t_end
        self._acc_tol = acc_tol

    @property
    def description(self) -> str:
        return f"conservation/{self._label}"

    def check(self) -> None:
        state = self._run_fn(self._dt, self._t_end)
        exact = self._exact_fn(state.t)
        err = max(abs(float(state.u[i]) - exact[i]) for i in range(self._n))
        assert (
            err < self._acc_tol
        ), f"{self._label}: max abundance error {err:.2e} > {self._acc_tol:.2e}"
        assert_conservation(state.u, self._n, label=self._label)


# ---------------------------------------------------------------------------
# B-series order claims
# ---------------------------------------------------------------------------


class _RKOrderClaim(Claim):
    """Verify B-series order conditions: α(τ) = 1/γ(τ) for all trees |τ| ≤ p."""

    def __init__(
        self, instance: RungeKuttaIntegrator | ImplicitRungeKuttaIntegrator, label: str
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


# ---------------------------------------------------------------------------
# Stability claims (DIRK)
# ---------------------------------------------------------------------------


class _AStabilityClaim(Claim):
    """Verify A-stability: |R(iω)| ≤ 1 for sampled imaginary-axis points."""

    def __init__(self, instance: ImplicitRungeKuttaIntegrator, label: str) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"a_stability/{self._label}"

    def check(self) -> None:
        R_expr = stability_function(self._instance.A_sym, self._instance.b_sym)
        z = sympy.Symbol("z")
        R_func = sympy.lambdify(z, R_expr, modules="cmath")
        for omega in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            R_val = abs(R_func(complex(0.0, omega)))
            assert R_val <= 1.0 + 1e-10, (
                f"{self._label}: |R(i·{omega})| = {R_val:.8f} > 1 "
                "(A-stability violated)"
            )


class _LStabilityClaim(Claim):
    """Verify L-stability: |R(z)| → 0 as Re(z) → −∞."""

    def __init__(self, instance: ImplicitRungeKuttaIntegrator, label: str) -> None:
        self._instance = instance
        self._label = label

    @property
    def description(self) -> str:
        return f"l_stability/{self._label}"

    def check(self) -> None:
        R_expr = stability_function(self._instance.A_sym, self._instance.b_sym)
        z = sympy.Symbol("z")
        R_func = sympy.lambdify(z, R_expr, modules="cmath")
        R_inf = abs(R_func(-1e6))
        assert (
            R_inf < 1e-4
        ), f"{self._label}: |R(−1e6)| = {R_inf:.2e}; method is not L-stable"


# ---------------------------------------------------------------------------
# PI controller claims
# ---------------------------------------------------------------------------


class _StepperClaim(Claim):
    """Verify end-to-end TimeStepper.advance accuracy at fixed step size."""

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
        final = TimeStepper(self._instance, controller=ConstantStep(self._dt)).advance(
            BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), 0.0, self._t_end
        )
        assert (
            abs(final.t - self._t_end) < 1e-12
        ), f"{self._label}: final time {final.t} ≠ t_end {self._t_end}"
        assert (
            max_base_network_error(final.u, self._t_end) < self._rtol
        ), f"{self._label}: error > rtol {self._rtol}"
        assert_conservation(final.u, 2, label=self._label)


class _PIAccuracyClaim(Claim):
    """Verify PIController achieves error ≤ c_rel · tol on the decay network."""

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
        p = self._instance.order
        pi = PIController(alpha=PI_ALPHA / p, beta=PI_BETA / p, tol=self._tol, dt0=0.1)
        final = TimeStepper(self._instance, controller=pi).advance(
            BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), 0.0, self._t_end
        )
        err = max_base_network_error(final.u, self._t_end)
        assert (
            err <= self._c_rel * self._tol
        ), f"{self._label}: error {err:.2e} > {self._c_rel} × tol {self._tol:.2e}"
        assert_conservation(final.u, 2, label=self._label)


class _PIWorkPrecisionClaim(Claim):
    """Verify that error decreases as tol decreases (adaptive control is working)."""

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
        p = self._instance.order

        def _run(tol: float) -> float:
            pi = PIController(alpha=PI_ALPHA / p, beta=PI_BETA / p, tol=tol, dt0=0.1)
            final = TimeStepper(self._instance, controller=pi).advance(
                BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), 0.0, self._t_end
            )
            assert_conservation(final.u, 2, label=self._label)
            return max_base_network_error(final.u, self._t_end)

        err_coarse, err_fine = _run(self._tol_coarse), _run(self._tol_fine)
        assert err_fine < err_coarse, (
            f"{self._label}: tighter tol did not reduce error "
            f"(coarse={err_coarse:.2e}, fine={err_fine:.2e})"
        )


# ---------------------------------------------------------------------------
# Nordsieck claims
# ---------------------------------------------------------------------------


class _NordsieckRoundTripClaim(Claim):
    """Verify pure Nordsieck order-change and step-size transforms preserve state."""

    @property
    def description(self) -> str:
        return "nordsieck_round_trip/order_and_step_size"

    def check(self) -> None:
        nh = NordsieckHistory(
            h=0.1,
            z=(
                Tensor([0.8, 0.2]),
                Tensor([-0.08, 0.08]),
                Tensor([0.004, -0.004]),
            ),
        )
        raised = nh.change_order(4)
        lowered = raised.change_order(2)
        assert lowered.q == nh.q and lowered.h == nh.h
        for lhs, rhs in zip(lowered.z, nh.z, strict=True):
            assert float(norm(lhs - rhs)) == 0.0
        assert raised.q == 4
        assert float(norm(raised.z[0] - nh.z[0])) == 0.0
        assert float(norm(raised.z[3])) == 0.0 and float(norm(raised.z[4])) == 0.0
        rescaled = nh.rescale_step(0.05).rescale_step(0.1)
        assert rescaled.h == nh.h
        for lhs, rhs in zip(rescaled.z, nh.z, strict=True):
            assert float(norm(lhs - rhs)) < 1e-16


class _NordsieckRescaledAccuracyClaim(Claim):
    """Verify transformed Nordsieck states retain fixed-order accuracy."""

    def __init__(
        self,
        instance: MultistepIntegrator,
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
        _k = 1.0  # base_network decay rate

        z_terms = [Tensor([1.0, 0.0])]
        for j in range(1, self._q_source + 1):
            deriv = (-_k) ** j
            scale = (self._h_source**j) / math.factorial(j)
            z_terms.append(Tensor([scale * deriv, -scale * deriv]))
        nh = (
            NordsieckHistory(self._h_source, tuple(z_terms))
            .change_order(self._q_target)
            .rescale_step(h)
        )
        state = ODEState(0.0, nh.z[0], h, 0.0, nh)
        for _ in range(round((t_end - state.t) / h)):
            state = inst.step(self._rhs, state, h)
        transformed_err = max_base_network_error(state.u, state.t)

        fresh = inst.init_state(self._rhs, 0.0, Tensor([1.0, 0.0]), h)
        for _ in range(round((t_end - fresh.t) / h)):
            fresh = inst.step(self._rhs, fresh, h)
        fresh_err = max_base_network_error(fresh.u, fresh.t)

        assert transformed_err <= 2.0 * fresh_err, (
            f"{self._label}: transformed error {transformed_err:.3e} "
            f"> 2× fresh-init error {fresh_err:.3e}"
        )


# ---------------------------------------------------------------------------
# Variable-order claims
# ---------------------------------------------------------------------------


class _VariableOrderClimbClaim(Claim):
    """Verify variable-order Adams climbs to max order on a smooth network."""

    @property
    def description(self) -> str:
        return "variable_order/climbs_on_smooth_network"

    def check(self) -> None:
        selector = OrderSelector(q_min=2, q_max=4, atol=1e-4, rtol=1e-4)
        inst = VariableOrderNordsieckIntegrator(adams_family, selector)
        state = inst.advance(
            BlackBoxRHS(decay_f), Tensor([1.0, 0.0, 0.0]), t0=0.0, t_end=1.5, dt0=0.025
        )
        assert max(inst.accepted_orders) == 4 and inst.accepted_orders[-1] == 4
        x0, x1, x2 = decay_exact(state.t)
        err = max(abs(float(state.u[i]) - v) for i, v in enumerate((x0, x1, x2)))
        assert err < 5e-4
        assert_conservation(state.u, 3, label="variable_order/climb")


class _VariableOrderDropClaim(Claim):
    """Verify variable-order Adams lowers order when the network sharpens."""

    @property
    def description(self) -> str:
        return "variable_order/drops_on_sharpening_network"

    def check(self) -> None:
        selector = OrderSelector(q_min=2, q_max=4, atol=1e-4, rtol=1e-4)
        inst = VariableOrderNordsieckIntegrator(adams_family, selector, q_initial=4)
        state = inst.advance(
            BlackBoxRHS(sharpening_decay_f),
            Tensor([1.0, 0.0, 0.0]),
            t0=0.0,
            t_end=1.0,
            dt0=0.02,
        )
        post_sharpen = [
            q
            for t, q in zip(inst.accepted_times, inst.accepted_orders, strict=True)
            if t > 0.55
        ]
        assert post_sharpen and min(post_sharpen) == 2
        assert_conservation(state.u, 3, label="variable_order/drop")
        for i in range(3):
            assert float(state.u[i]) >= -1e-10


# ---------------------------------------------------------------------------
# Stiffness detection and family-switching claims
# ---------------------------------------------------------------------------


class _StiffnessDiagnosticClaim(Claim):
    """Verify the Gershgorin stiffness estimate crosses the expected threshold."""

    @property
    def description(self) -> str:
        return "stiffness/diagnostic_threshold"

    def check(self) -> None:
        diag = StiffnessDiagnostic()
        jac = Tensor([[-3.0, 1.0], [0.5, -2.0]])
        below = diag.update(jac, h=0.1)
        above = diag.update(jac, h=0.3)
        assert abs(below - 0.4) < 1e-15 and abs(above - 1.2) < 1e-15
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
        z = (
            Tensor([0.8, 0.15, 0.05]),
            Tensor([-0.02, 0.01, 0.01]),
            Tensor([0.0, 0.0, 0.0]),
        )
        state = ODEState(0.25, z[0], 0.05, 0.0, NordsieckHistory(0.05, z))
        bdf_state = inst.transform_family(state, "bdf")
        round_trip = inst.transform_family(bdf_state, "adams")
        assert nordsieck_solution_distance(state, round_trip) == 0.0
        assert round_trip.history.h == state.history.h
        assert round_trip.history.q == state.history.q


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
            STIFFENING_RHS, Tensor([1.0, 0.0, 0.0]), t0=0.0, t_end=0.8, dt=0.02
        )
        pre = [
            f
            for t, f in zip(inst.accepted_times, inst.accepted_families, strict=True)
            if t < 0.45
        ]
        post = [
            f
            for t, f in zip(inst.accepted_times, inst.accepted_families, strict=True)
            if t > 0.55
        ]
        assert pre and post
        assert set(pre) == {"adams"} and "bdf" in post
        assert inst.switch_count >= 1 and max(inst.accepted_stiffness) > 1.0
        assert_conservation(state.u, 3, label="stiffness/switch")
        for i in range(3):
            assert float(state.u[i]) >= -1e-10


# ---------------------------------------------------------------------------
# VODE controller claim
# ---------------------------------------------------------------------------


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
                stiff_threshold=1.0, nonstiff_threshold=0.4
            ),
            q_initial=2,
            initial_family="adams",
        )
        state = controller.advance(
            VODE_RHS, Tensor([1.0, 0.0, 0.0]), t0=0.0, t_end=0.7, dt0=0.005
        )
        early = [
            f
            for t, f in zip(
                controller.accepted_times, controller.accepted_families, strict=True
            )
            if t < 0.35
        ]
        late = [
            f
            for t, f in zip(
                controller.accepted_times, controller.accepted_families, strict=True
            )
            if t > 0.5
        ]
        assert early and late
        assert set(early) == {"adams"} and "bdf" in late
        assert (
            controller.family_switches >= 1 and max(controller.accepted_stiffness) > 1.0
        )
        assert max(controller.accepted_orders) > 2
        assert_conservation(state.u, 3, label="vode/switch")
        for i in range(3):
            assert float(state.u[i]) >= -1e-10


# ---------------------------------------------------------------------------
# Exponential integrator phi-function claim
# ---------------------------------------------------------------------------


class _PhiFunctionClaim(Claim):
    """Verify phi-function coefficient algebra on a nilpotent matrix."""

    @property
    def description(self) -> str:
        return "exponential/phi_function_coefficients"

    def check(self) -> None:
        A = Tensor([[0.0, 1.0], [0.0, 0.0]])
        v = Tensor([0.0, 1.0])
        expected = [
            Tensor([1.0, 1.0]),
            Tensor([0.5, 1.0]),
            Tensor([1.0 / 6.0, 0.5]),
            Tensor([1.0 / 24.0, 1.0 / 6.0]),
        ]
        for k, exp in enumerate(expected):
            phi = PhiFunction(k).apply(A, v)
            assert float(norm(phi - exp)) < 1e-14, (
                f"phi_{k}: got {[float(phi[i]) for i in range(2)]}, "
                f"expected {[float(exp[i]) for i in range(2)]}"
            )


# ---------------------------------------------------------------------------
# AutoDispatch claims
# ---------------------------------------------------------------------------


class _AutoDispatchClaim(Claim):
    """Verify AutoIntegrator dispatches to the right specialist integrator."""

    def __init__(
        self,
        label: str,
        rhs: Any,
        state: ODEState,
        dt: float,
        direct: TimeIntegrator,
        expected_order: int,
    ) -> None:
        self._label = label
        self._rhs = rhs
        self._state = state
        self._dt = dt
        self._direct = direct
        self._expected_order = expected_order

    @property
    def description(self) -> str:
        return f"auto_dispatch/{self._label}"

    def check(self) -> None:
        auto = AutoIntegrator()
        expected = self._direct.step(self._rhs, self._state, self._dt)
        actual = auto.step(self._rhs, self._state, self._dt)
        assert self._direct.order == self._expected_order
        assert actual.t == expected.t and actual.dt == expected.dt
        assert actual.err == expected.err
        assert (
            float(norm(actual.u - expected.u)) < 1e-15
        ), f"{self._label}: AutoIntegrator did not match direct branch"


# ---------------------------------------------------------------------------
# Type coherence and Newton kernel claims
# ---------------------------------------------------------------------------


class _TypeCoherenceClaim(Claim):
    """Verify a specialist integrator class inherits TimeIntegrator."""

    def __init__(self, cls: type, instances: list[TimeIntegrator], label: str) -> None:
        self._cls = cls
        self._instances = instances
        self._label = label

    @property
    def description(self) -> str:
        return f"type_coherence/{self._label}"

    def check(self) -> None:
        assert issubclass(
            self._cls, TimeIntegrator
        ), f"{self._label}: {self._cls.__name__} must inherit TimeIntegrator"
        for inst in self._instances:
            assert isinstance(
                inst, TimeIntegrator
            ), f"{self._label}: {inst!r} must be an instance of TimeIntegrator"


class _NewtonSolveClaim(Claim):
    """Verify newton_solve converges to the exact root on a specific problem."""

    def __init__(
        self,
        label: str,
        y_exp_vals: list[float],
        gamma_dt: float,
        f_vals: Callable[[list[float]], list[float]],
        jac_vals: Callable[[list[float]], list[list[float]]],
        y_star: list[float],
        tol: float = 1e-11,
    ) -> None:
        self._label = label
        self._y_exp_vals = y_exp_vals
        self._gamma_dt = gamma_dt
        self._f_vals = f_vals
        self._jac_vals = jac_vals
        self._y_star = y_star
        self._tol = tol

    @property
    def description(self) -> str:
        return f"newton_solve/{self._label}"

    def check(self) -> None:
        backend = Tensor([0.0]).backend
        n = len(self._y_exp_vals)

        def f(y: Tensor) -> Tensor:
            vals = self._f_vals([float(y[i]) for i in range(n)])
            return Tensor(vals, backend=backend)

        def jac(y: Tensor) -> Tensor:
            return Tensor(
                self._jac_vals([float(y[i]) for i in range(n)]), backend=backend
            )

        y_exp = Tensor(self._y_exp_vals, backend=backend)
        y_sol = newton_solve(y_exp, gamma_dt=self._gamma_dt, f=f, jac=jac)
        for i, expected in enumerate(self._y_star):
            actual = float(y_sol[i])
            assert (
                abs(actual - expected) < self._tol
            ), f"{self._label}: y[{i}] = {actual:.15f}; expected {expected:.15f}"


# ---------------------------------------------------------------------------
# Reaction network RHS claims
# ---------------------------------------------------------------------------
#
# A⇌B with equal forward/reverse rate k.
# S = [[-1], [+1]], conservation: X_A + X_B = const.
# Exact: X_A(t) = 0.5 + 0.5·exp(−2k·t), X_B(t) = 1 − X_A.

_RN_K = 1.0
_RN_S = Tensor([[-1.0], [1.0]])
_RN_U0 = Tensor([1.0, 0.0])


def _rn_r_plus(t: float, u: Tensor) -> Tensor:
    return Tensor([_RN_K * float(u[0])], backend=u.backend)


def _rn_r_minus(t: float, u: Tensor) -> Tensor:
    return Tensor([_RN_K * float(u[1])], backend=u.backend)


def _rn_exact(t: float) -> tuple[float, float]:
    xa = 0.5 + 0.5 * math.exp(-2.0 * _RN_K * t)
    return xa, 1.0 - xa


class _NetworkInvariantsClaim(Claim):
    """Verify structural invariants of ReactionNetworkRHS on the A⇌B toy."""

    @property
    def description(self) -> str:
        return "reaction_network/invariants"

    def check(self) -> None:
        rhs = ReactionNetworkRHS(
            stoichiometry_matrix=_RN_S,
            forward_rate=_rn_r_plus,
            reverse_rate=_rn_r_minus,
            initial_state=_RN_U0,
        )
        # stoichiometry_matrix round-trips
        for i in range(2):
            assert float(rhs.stoichiometry_matrix[i, 0]) == pytest.approx(
                float(_RN_S[i, 0])
            )
        # conservation_basis: shape (1, 2), row proportional to [1, 1]
        n_conserved, n_species = rhs.conservation_basis.shape
        assert n_conserved == 1 and n_species == 2
        row = [float(rhs.conservation_basis[0, j]) for j in range(2)]
        ratio = row[0] / row[1] if abs(row[1]) > 1e-14 else None
        assert ratio is not None and abs(ratio - 1.0) < 1e-12
        # conservation_targets
        target = float(rhs.conservation_targets[0])
        expected = row[0] * float(_RN_U0[0]) + row[1] * float(_RN_U0[1])
        assert abs(target - expected) < 1e-12
        # __call__ = S·(r⁺ − r⁻)
        u_test = Tensor([0.7, 0.3])
        f_actual = rhs(0.0, u_test)
        r_net = _RN_K * (float(u_test[0]) - float(u_test[1]))
        assert float(f_actual[0]) == pytest.approx(-r_net, abs=1e-12)
        assert float(f_actual[1]) == pytest.approx(r_net, abs=1e-12)
        # detailed balance at equilibrium
        f_eq = rhs(0.0, Tensor([0.5, 0.5]))
        assert abs(float(f_eq[0])) < 1e-14 and abs(float(f_eq[1])) < 1e-14


# ---------------------------------------------------------------------------
# Conservation projection claims
# ---------------------------------------------------------------------------

_PROJ_BASIS = Tensor([[1.0, 1.0, 1.0]])
_PROJ_TARGETS = Tensor([1.0])


class _ProjectConservedClaim(Claim):
    """Verify idempotence, minimum-norm, and round-trip of project_conserved."""

    @property
    def description(self) -> str:
        return "project_conserved/invariants"

    def check(self) -> None:
        # Idempotence
        u_off = Tensor([0.4, 0.4, 0.4])
        pu = project_conserved(u_off, _PROJ_BASIS, _PROJ_TARGETS)
        ppu = project_conserved(pu, _PROJ_BASIS, _PROJ_TARGETS)
        for i in range(3):
            assert abs(float(ppu[i]) - float(pu[i])) < 1e-12
        # Round-trip
        u_on = Tensor([0.5, 0.3, 0.2])
        pu_on = project_conserved(u_on, _PROJ_BASIS, _PROJ_TARGETS)
        for i in range(3):
            assert abs(float(pu_on[i]) - float(u_on[i])) < 1e-12
        # Minimum-norm
        u_far = Tensor([0.5, 0.5, 0.5])
        pu_far = project_conserved(u_far, _PROJ_BASIS, _PROJ_TARGETS)
        dist_proj = (
            sum((float(pu_far[i]) - float(u_far[i])) ** 2 for i in range(3)) ** 0.5
        )
        dist_arb = (
            sum(
                (float(Tensor([1.0, 0.0, 0.0])[i]) - float(u_far[i])) ** 2
                for i in range(3)
            )
            ** 0.5
        )
        assert dist_proj <= dist_arb


# ---------------------------------------------------------------------------
# Projected Newton claims
# ---------------------------------------------------------------------------

_PN_CONSTRAINT = Tensor([[1.0, 1.0]])


class _ProjectedNewtonClaim(Claim):
    """Projected Newton: result on manifold, agrees with exact 1D solution."""

    def __init__(
        self, label: str, k: float, gamma_dt: float, y_exp: list[float]
    ) -> None:
        self._label = label
        self._k = k
        self._gamma_dt = gamma_dt
        self._y_exp = y_exp

    @property
    def description(self) -> str:
        return f"projected_newton/{self._label}"

    def check(self) -> None:
        k, gdt = self._k, self._gamma_dt
        y_exp = Tensor(self._y_exp)

        def f(y: Tensor) -> Tensor:
            return Tensor([-k * float(y[0]), k * float(y[0])], backend=y.backend)

        def jac(y: Tensor) -> Tensor:
            return Tensor([[-k, 0.0], [k, 0.0]], backend=y.backend)

        result = newton_solve(y_exp, gdt, f, jac, constraint_gradients=_PN_CONSTRAINT)
        c_res = abs(float(result[0]) + float(result[1]) - 1.0)
        assert c_res < 1e-12, f"{self._label}: constraint residual {c_res:.2e}"
        y0_exact = self._y_exp[0] / (1.0 + gdt * k)
        assert abs(float(result[0]) - y0_exact) < 1e-10
        assert abs(float(result[1]) - (1.0 - y0_exact)) < 1e-10


# ---------------------------------------------------------------------------
# Constraint lifecycle claims
# ---------------------------------------------------------------------------
#
# A⇌B with k_f=1.0, k_r=0.5.  Conservation: A+B=1.  Equilibrium: A_eq=1/3.
# Exact: A(t) = 1/3 + 2/3·exp(−1.5t).

_LC_K_F = 1.0
_LC_K_R = 0.5
_LC_S = Tensor([[-1.0], [1.0]])
_LC_U0 = Tensor([1.0, 0.0])


def _lc_r_plus(t: float, u: Tensor) -> Tensor:
    return Tensor([_LC_K_F * float(u[0])], backend=u.backend)


def _lc_r_minus(t: float, u: Tensor) -> Tensor:
    return Tensor([_LC_K_R * float(u[1])], backend=u.backend)


def _lc_make_rhs() -> ReactionNetworkRHS:
    return ReactionNetworkRHS(_LC_S, _lc_r_plus, _lc_r_minus, _LC_U0)


def _lc_make_ctrl(rhs: ReactionNetworkRHS) -> ConstraintAwareController:
    return ConstraintAwareController(
        rhs=rhs,
        integrator=implicit_midpoint,
        inner=PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=0.05),
        eps_activate=0.01,
        eps_deactivate=0.1,
    )


class _ActivationClaim(Claim):
    """Constraint pair 0 must be active at t=6."""

    @property
    def description(self) -> str:
        return "constraint_lifecycle/activation"

    def check(self) -> None:
        rhs = _lc_make_rhs()
        ctrl = _lc_make_ctrl(rhs)
        state = ctrl.advance(_LC_U0, 0.0, 6.0)
        assert state.active_constraints == frozenset({0})
        assert ctrl.activation_events


class _NoChatteringClaim(Claim):
    """No deactivation events during run to t=6."""

    @property
    def description(self) -> str:
        return "constraint_lifecycle/no_chattering"

    def check(self) -> None:
        rhs = _lc_make_rhs()
        ctrl = _lc_make_ctrl(rhs)
        ctrl.advance(_LC_U0, 0.0, 6.0)
        assert (
            not ctrl.deactivation_events
        ), f"deactivation events (chattering): {ctrl.deactivation_events}"


class _ConsistentInitClaim(Claim):
    """Consistent initialization lands state on constraint manifold."""

    @property
    def description(self) -> str:
        return "constraint_lifecycle/consistent_init"

    def check(self) -> None:
        rhs = _lc_make_rhs()
        ctrl = _lc_make_ctrl(rhs)
        state = ctrl.advance(_LC_U0, 0.0, 6.0)
        rp = float(rhs.forward_rate(state.t, state.u)[0])
        rm = float(rhs.reverse_rate(state.t, state.u)[0])
        ratio = abs(rp - rm) / max(abs(rp), abs(rm), 1e-100)
        assert ratio < 0.01, f"final equilibrium ratio {ratio:.3e} ≥ eps_activate"


class _DeactivationClaim(Claim):
    """Constraint deactivates when state is far from equilibrium."""

    @property
    def description(self) -> str:
        return "constraint_lifecycle/deactivation"

    def check(self) -> None:
        rhs = _lc_make_rhs()
        ctrl = _lc_make_ctrl(rhs)
        u_far = Tensor([0.7, 0.3])
        state = ctrl.advance(u_far, 0.0, 0.5, initial_active=frozenset({0}))
        assert ctrl.deactivation_events
        assert state.active_constraints == frozenset()


# ---------------------------------------------------------------------------
# NSE solver claims  (A⇌B⇌C)
# ---------------------------------------------------------------------------

_NSE_K = 1.0
_NSE_S = Tensor([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]])
_NSE_U0 = Tensor([1.0, 0.0, 0.0])


def _nse_r_plus(t: float, u: Tensor) -> Tensor:
    return Tensor([_NSE_K * float(u[0]), _NSE_K * float(u[1])], backend=u.backend)


def _nse_r_minus(t: float, u: Tensor) -> Tensor:
    return Tensor([_NSE_K * float(u[1]), _NSE_K * float(u[2])], backend=u.backend)


class _NSEDirectSolveClaim(Claim):
    """solve_nse(rhs, u_near, t=0) recovers A=B=C=1/3 to 1e-10."""

    @property
    def description(self) -> str:
        return "nse_solver/direct_solve"

    def check(self) -> None:
        rhs = ReactionNetworkRHS(_NSE_S, _nse_r_plus, _nse_r_minus, _NSE_U0)
        result = solve_nse(rhs, Tensor([0.34, 0.33, 0.33]), t=0.0)
        for i in range(3):
            xi = float(result[i])
            assert (
                abs(xi - 1.0 / 3.0) < 1e-10
            ), f"species {i}: |u[{i}] - 1/3| = {abs(xi - 1/3):.3e}"


class _NonlinearSolveScalarClaim(Claim):
    """nonlinear_solve finds √2 for the scalar problem x²=2."""

    @property
    def description(self) -> str:
        return "nse_solver/nonlinear_solve_scalar"

    def check(self) -> None:
        def F(x: Tensor) -> Tensor:
            return Tensor([float(x[0]) ** 2 - 2.0], backend=x.backend)

        def J(x: Tensor) -> Tensor:
            return Tensor([[2.0 * float(x[0])]], backend=x.backend)

        root = nonlinear_solve(F, J, Tensor([1.0]))
        assert (
            abs(float(root[0]) - 2.0**0.5) < 1e-12
        ), f"root = {float(root[0]):.15f}, expected √2"


# ---------------------------------------------------------------------------
# Rate-threshold guard  (10-species chain)
# ---------------------------------------------------------------------------

_CHAIN_RT_S_ROWS = [[0.0] * 9 for _ in range(10)]
for _j in range(9):
    _CHAIN_RT_S_ROWS[_j][_j] = -1.0
    _CHAIN_RT_S_ROWS[_j + 1][_j] = 1.0
_CHAIN_RT_S = Tensor(_CHAIN_RT_S_ROWS)
_CHAIN_RT_U0 = Tensor([1.0] + [0.0] * 9)
_CHAIN_RT_K = 1.0


def _chain_rt_r_plus(t: float, u: Tensor) -> Tensor:
    return Tensor([_CHAIN_RT_K * float(u[j]) for j in range(9)], backend=u.backend)


def _chain_rt_r_minus(t: float, u: Tensor) -> Tensor:
    return Tensor([_CHAIN_RT_K * float(u[j + 1]) for j in range(9)], backend=u.backend)


class _RateThresholdClaim(Claim):
    """Absent-species pairs return inf ratio — no spurious early activation."""

    @property
    def description(self) -> str:
        return "rate_threshold/absent_pairs"

    def check(self) -> None:
        rhs = ReactionNetworkRHS(
            _CHAIN_RT_S, _chain_rt_r_plus, _chain_rt_r_minus, _CHAIN_RT_U0
        )
        ctrl = ConstraintAwareController(
            rhs=rhs,
            integrator=implicit_midpoint,
            inner=PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=0.01),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        ctrl.advance(_CHAIN_RT_U0, 0.0, 0.05)
        for t_ev, pairs in ctrl.activation_events:
            for j in pairs:
                assert (
                    j == 0 or t_ev > 0.04
                ), f"spurious early activation of pair {j} at t={t_ev:.4f}"


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

# ---------------------------------------------------------------------------
# Symplectic convergence helper — defined here so it's in scope for the list.
# ---------------------------------------------------------------------------


def _run_symplectic(inst: SymplecticCompositionIntegrator, dt: float) -> float:
    H = HamiltonianRHS(dT_dp=lambda p: p, dV_dq=lambda q: q, split_index=1)
    state = ODEState(0.0, Tensor([1.0, 0.0]))
    for _ in range(round(1.0 / dt)):
        state = inst.step(H, state, dt)
    return math.sqrt(
        (float(state.u[0]) - math.cos(state.t)) ** 2
        + (float(state.u[1]) - (-math.sin(state.t))) ** 2
    )


# One _ConvergenceClaim per integrator/family.  Run functions are closures that
# capture the integrator instance and RHS; DT_BASE / order and BDF_N_HALVINGS
# give the correct dt range for multistep bootstrap clearing.

_CONVERGENCE_CLAIMS: list[_ConvergenceClaim] = [
    # Explicit RK — 3-species decay chain, analytical exact
    _ConvergenceClaim(
        "rk/forward_euler",
        lambda dt: max_decay_error(
            run_rk(forward_euler, BlackBoxRHS(decay_f), Tensor([1.0, 0.0, 0.0]), dt).u,
            1.0,
        ),
        forward_euler.order,
    ),
    _ConvergenceClaim(
        "rk/midpoint",
        lambda dt: max_decay_error(
            run_rk(midpoint, BlackBoxRHS(decay_f), Tensor([1.0, 0.0, 0.0]), dt).u, 1.0
        ),
        midpoint.order,
    ),
    _ConvergenceClaim(
        "rk/heun",
        lambda dt: max_decay_error(
            run_rk(heun, BlackBoxRHS(decay_f), Tensor([1.0, 0.0, 0.0]), dt).u, 1.0
        ),
        heun.order,
    ),
    _ConvergenceClaim(
        "rk/ralston",
        lambda dt: max_decay_error(
            run_rk(ralston, BlackBoxRHS(decay_f), Tensor([1.0, 0.0, 0.0]), dt).u, 1.0
        ),
        ralston.order,
    ),
    _ConvergenceClaim(
        "rk/rk4",
        lambda dt: max_decay_error(
            run_rk(rk4, BlackBoxRHS(decay_f), Tensor([1.0, 0.0, 0.0]), dt).u, 1.0
        ),
        rk4.order,
    ),
    _ConvergenceClaim(
        "rk/dormand_prince",
        lambda dt: max_decay_error(
            run_rk(dormand_prince, BlackBoxRHS(decay_f), Tensor([1.0, 0.0, 0.0]), dt).u,
            1.0,
        ),
        dormand_prince.order,
    ),
    _ConvergenceClaim(
        "rk/bogacki_shampine",
        lambda dt: max_decay_error(
            run_rk(
                bogacki_shampine, BlackBoxRHS(decay_f), Tensor([1.0, 0.0, 0.0]), dt
            ).u,
            1.0,
        ),
        bogacki_shampine.order,
    ),
    # DIRK — 2-species base network, analytical exact
    _ConvergenceClaim(
        "dirk/backward_euler",
        lambda dt: max_base_network_error(
            run_rk(backward_euler, BASE_RHS, Tensor([1.0, 0.0]), dt).u, 1.0
        ),
        backward_euler.order,
    ),
    _ConvergenceClaim(
        "dirk/implicit_midpoint",
        lambda dt: max_base_network_error(
            run_rk(implicit_midpoint, BASE_RHS, Tensor([1.0, 0.0]), dt).u, 1.0
        ),
        implicit_midpoint.order,
    ),
    _ConvergenceClaim(
        "dirk/crouzeix_3",
        lambda dt: max_base_network_error(
            run_rk(crouzeix_3, BASE_RHS, Tensor([1.0, 0.0]), dt).u, 1.0
        ),
        crouzeix_3.order,
    ),
    # IMEX — 2-species IMEX split, analytical exact
    _ConvergenceClaim(
        "imex/ars222",
        lambda dt: max_base_network_error(
            run_rk(ars222, BASE_RHS_IMEX, Tensor([1.0, 0.0]), dt).u, 1.0
        ),
        ars222.order,
    ),
    # Adams-Bashforth — 2-species base network, analytical exact
    _ConvergenceClaim(
        "ab/ab2",
        lambda dt: max_base_network_error(
            run_rk(ab2, BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), dt).u, 1.0
        ),
        ab2.order,
    ),
    _ConvergenceClaim(
        "ab/ab3",
        lambda dt: max_base_network_error(
            run_rk(ab3, BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), dt).u, 1.0
        ),
        ab3.order,
    ),
    _ConvergenceClaim(
        "ab/ab4",
        lambda dt: max_base_network_error(
            run_rk(ab4, BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), dt).u, 1.0
        ),
        ab4.order,
    ),
    # BDF — 2-species base network via JacobianRHS, dt_base / order
    _ConvergenceClaim(
        "bdf/bdf1",
        lambda dt: (lambda s: max_base_network_error(s.u, s.t))(
            run_multistep(bdf1, BASE_RHS, Tensor([1.0, 0.0]), dt)
        ),
        bdf1.order,
        dt_base=DT_BASE / bdf1.order,
        n_halvings=BDF_N_HALVINGS - (math.floor(math.log2(bdf1.order)) + 1),
    ),
    _ConvergenceClaim(
        "bdf/bdf2",
        lambda dt: (lambda s: max_base_network_error(s.u, s.t))(
            run_multistep(bdf2, BASE_RHS, Tensor([1.0, 0.0]), dt)
        ),
        bdf2.order,
        dt_base=DT_BASE / bdf2.order,
        n_halvings=BDF_N_HALVINGS - (math.floor(math.log2(bdf2.order)) + 1),
    ),
    _ConvergenceClaim(
        "bdf/bdf3",
        lambda dt: (lambda s: max_base_network_error(s.u, s.t))(
            run_multistep(bdf3, BASE_RHS, Tensor([1.0, 0.0]), dt)
        ),
        bdf3.order,
        dt_base=DT_BASE / bdf3.order,
        n_halvings=BDF_N_HALVINGS - (math.floor(math.log2(bdf3.order)) + 1),
    ),
    _ConvergenceClaim(
        "bdf/bdf4",
        lambda dt: (lambda s: max_base_network_error(s.u, s.t))(
            run_multistep(bdf4, BASE_RHS, Tensor([1.0, 0.0]), dt)
        ),
        bdf4.order,
        dt_base=DT_BASE / bdf4.order,
        n_halvings=BDF_N_HALVINGS - (math.floor(math.log2(bdf4.order)) + 1),
    ),
    # Adams-Moulton — same as BDF but plain callable RHS
    _ConvergenceClaim(
        "am/am1",
        lambda dt: (lambda s: max_base_network_error(s.u, s.t))(
            run_multistep(
                adams_moulton1, BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), dt
            )
        ),
        adams_moulton1.order,
        dt_base=DT_BASE / adams_moulton1.order,
        n_halvings=BDF_N_HALVINGS - (math.floor(math.log2(adams_moulton1.order)) + 1),
    ),
    _ConvergenceClaim(
        "am/am2",
        lambda dt: (lambda s: max_base_network_error(s.u, s.t))(
            run_multistep(
                adams_moulton2, BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), dt
            )
        ),
        adams_moulton2.order,
        dt_base=DT_BASE / adams_moulton2.order,
        n_halvings=BDF_N_HALVINGS - (math.floor(math.log2(adams_moulton2.order)) + 1),
    ),
    _ConvergenceClaim(
        "am/am3",
        lambda dt: (lambda s: max_base_network_error(s.u, s.t))(
            run_multistep(
                adams_moulton3, BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), dt
            )
        ),
        adams_moulton3.order,
        dt_base=DT_BASE / adams_moulton3.order,
        n_halvings=BDF_N_HALVINGS - (math.floor(math.log2(adams_moulton3.order)) + 1),
    ),
    _ConvergenceClaim(
        "am/am4",
        lambda dt: (lambda s: max_base_network_error(s.u, s.t))(
            run_multistep(
                adams_moulton4, BlackBoxRHS(base_network_f), Tensor([1.0, 0.0]), dt
            )
        ),
        adams_moulton4.order,
        dt_base=DT_BASE / adams_moulton4.order,
        n_halvings=BDF_N_HALVINGS - (math.floor(math.log2(adams_moulton4.order)) + 1),
    ),
    # ETD — reference-solution based, tolerance 0.25
    _ConvergenceClaim(
        "etd/etd_euler",
        (lambda ref: lambda dt: float(norm(integrate_etd(etd_euler, dt).u - ref.u)))(
            integrate_etd(etd_euler, 0.0015625)
        ),
        1,
        dt_base=0.025,
        n_halvings=2,
        tolerance=0.25,
    ),
    _ConvergenceClaim(
        "etd/etdrk2",
        (lambda ref: lambda dt: float(norm(integrate_etd(etdrk2, dt).u - ref.u)))(
            integrate_etd(etdrk2, 0.0015625)
        ),
        2,
        dt_base=0.025,
        n_halvings=2,
        tolerance=0.25,
    ),
    _ConvergenceClaim(
        "etd/cox_matthews_etdrk4",
        (
            lambda ref: lambda dt: float(
                norm(integrate_etd(cox_matthews_etdrk4, dt).u - ref.u)
            )
        )(integrate_etd(cox_matthews_etdrk4, 0.0015625)),
        4,
        dt_base=0.025,
        n_halvings=2,
        tolerance=0.25,
    ),
    _ConvergenceClaim(
        "etd/krogstad_etdrk4",
        (
            lambda ref: lambda dt: float(
                norm(integrate_etd(krogstad_etdrk4, dt).u - ref.u)
            )
        )(integrate_etd(krogstad_etdrk4, 0.0015625)),
        4,
        dt_base=0.025,
        n_halvings=2,
        tolerance=0.25,
    ),
    # Operator splitting — 2D oscillator, exact solution
    _ConvergenceClaim(
        "splitting/lie",
        lambda dt: float(
            norm(
                integrate_split(
                    CompositionIntegrator([rk4, rk4], lie_steps(), order=1), dt
                ).u
                - split_exact(1.0, Tensor([1.0, 0.0]))
            )
        ),
        1,
        dt_base=0.1,
        n_halvings=3,
        tolerance=0.25,
    ),
    _ConvergenceClaim(
        "splitting/strang",
        lambda dt: float(
            norm(
                integrate_split(
                    CompositionIntegrator([rk4, rk4], strang_steps(), order=2), dt
                ).u
                - split_exact(1.0, Tensor([1.0, 0.0]))
            )
        ),
        2,
        dt_base=0.1,
        n_halvings=3,
        tolerance=0.25,
    ),
    _ConvergenceClaim(
        "splitting/yoshida",
        lambda dt: float(
            norm(
                integrate_split(
                    CompositionIntegrator(
                        [forward_euler, forward_euler], yoshida_steps(), order=4
                    ),
                    dt,
                ).u
                - split_exact(1.0, Tensor([1.0, 0.0]))
            )
        ),
        4,
        dt_base=0.1,
        n_halvings=3,
        tolerance=0.25,
    ),
    # Symplectic — harmonic oscillator H = p²/2 + q²/2
    _ConvergenceClaim(
        "symplectic/symplectic_euler",
        lambda dt: _run_symplectic(symplectic_euler, dt),
        symplectic_euler.order,
    ),
    _ConvergenceClaim(
        "symplectic/leapfrog",
        lambda dt: _run_symplectic(leapfrog, dt),
        leapfrog.order,
    ),
    _ConvergenceClaim(
        "symplectic/forest_ruth",
        lambda dt: _run_symplectic(forest_ruth, dt),
        forest_ruth.order,
    ),
    _ConvergenceClaim(
        "symplectic/yoshida_6",
        lambda dt: _run_symplectic(yoshida_6, dt),
        yoshida_6.order,
        dt_base=0.25,
    ),
    _ConvergenceClaim(
        "symplectic/yoshida_8",
        lambda dt: _run_symplectic(yoshida_8, dt),
        yoshida_8.order,
        dt_base=0.25,
        n_halvings=3,
    ),
    # Reaction-network convergence — A⇌B
    _ConvergenceClaim(
        "reaction_network/forward_euler",
        lambda dt: max(
            abs(
                float(
                    run_rk(
                        forward_euler,
                        ReactionNetworkRHS(_RN_S, _rn_r_plus, _rn_r_minus, _RN_U0),
                        Tensor([1.0, 0.0]),
                        dt,
                    ).u[i]
                )
                - _rn_exact(1.0)[i]
            )
            for i in range(2)
        ),
        forward_euler.order,
    ),
    _ConvergenceClaim(
        "reaction_network/heun",
        lambda dt: max(
            abs(
                float(
                    run_rk(
                        heun,
                        ReactionNetworkRHS(_RN_S, _rn_r_plus, _rn_r_minus, _RN_U0),
                        Tensor([1.0, 0.0]),
                        dt,
                    ).u[i]
                )
                - _rn_exact(1.0)[i]
            )
            for i in range(2)
        ),
        heun.order,
    ),
    _ConvergenceClaim(
        "reaction_network/rk4",
        lambda dt: max(
            abs(
                float(
                    run_rk(
                        rk4,
                        ReactionNetworkRHS(_RN_S, _rn_r_plus, _rn_r_minus, _RN_U0),
                        Tensor([1.0, 0.0]),
                        dt,
                    ).u[i]
                )
                - _rn_exact(1.0)[i]
            )
            for i in range(2)
        ),
        rk4.order,
    ),
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

_A_STABILITY_CLAIMS: list[_AStabilityClaim] = [
    _AStabilityClaim(backward_euler, "backward_euler"),
    _AStabilityClaim(implicit_midpoint, "implicit_midpoint"),
    _AStabilityClaim(crouzeix_3, "crouzeix_3"),
]

_L_STABILITY_CLAIMS: list[_LStabilityClaim] = [
    _LStabilityClaim(backward_euler, "backward_euler"),
]

_CONSERVATION_CLAIMS: list[_ConservationClaim] = [
    # Explicit RK
    _ConservationClaim(
        "rk/rk4",
        lambda dt, t_end: run_conservation(
            rk4, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=1e-4,
    ),
    _ConservationClaim(
        "dirk/backward_euler",
        lambda dt, t_end: run_conservation(
            backward_euler, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=2e-2,
    ),
    _ConservationClaim(
        "dirk/implicit_midpoint",
        lambda dt, t_end: run_conservation(
            implicit_midpoint, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=1e-3,
    ),
    _ConservationClaim(
        "dirk/crouzeix_3",
        lambda dt, t_end: run_conservation(
            crouzeix_3, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=5e-4,
    ),
    # IMEX
    _ConservationClaim(
        "imex/ars222",
        lambda dt, t_end: run_conservation(
            ars222, DECAY_RHS_IMEX, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
    ),
    # Adams-Bashforth
    _ConservationClaim(
        "ab/ab2",
        lambda dt, t_end: run_conservation(
            ab2, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=5e-4,
    ),
    _ConservationClaim(
        "ab/ab3",
        lambda dt, t_end: run_conservation(
            ab3, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=5e-5,
    ),
    _ConservationClaim(
        "ab/ab4",
        lambda dt, t_end: run_conservation(
            ab4, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=5e-6,
    ),
    # BDF
    _ConservationClaim(
        "bdf/bdf1",
        lambda dt, t_end: run_multistep_conservation(
            bdf1, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=0.1,
    ),
    _ConservationClaim(
        "bdf/bdf2",
        lambda dt, t_end: run_multistep_conservation(
            bdf2, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=5e-3,
    ),
    _ConservationClaim(
        "bdf/bdf3",
        lambda dt, t_end: run_multistep_conservation(
            bdf3, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=5e-4,
    ),
    _ConservationClaim(
        "bdf/bdf4",
        lambda dt, t_end: run_multistep_conservation(
            bdf4, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=5e-5,
    ),
    # Adams-Moulton (uses plain callable — JacobianRHS satisfies RHSProtocol)
    _ConservationClaim(
        "am/am1",
        lambda dt, t_end: run_multistep_conservation(
            adams_moulton1, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=0.05,
    ),
    _ConservationClaim(
        "am/am2",
        lambda dt, t_end: run_multistep_conservation(
            adams_moulton2, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=5e-4,
    ),
    _ConservationClaim(
        "am/am3",
        lambda dt, t_end: run_multistep_conservation(
            adams_moulton3, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=1e-5,
    ),
    _ConservationClaim(
        "am/am4",
        lambda dt, t_end: run_multistep_conservation(
            adams_moulton4, DECAY_RHS, Tensor([1.0, 0.0, 0.0]), dt, t_end
        ),
        3,
        decay_exact,
        acc_tol=1e-6,
    ),
]

_NORDSIECK_ROUND_TRIP_CLAIMS: list[_NordsieckRoundTripClaim] = [
    _NordsieckRoundTripClaim(),
]

_NORDSIECK_RESCALED_ACCURACY_CLAIMS: list[_NordsieckRescaledAccuracyClaim] = [
    _NordsieckRescaledAccuracyClaim(bdf2, "bdf2", BASE_RHS),
    _NordsieckRescaledAccuracyClaim(adams_moulton2, "am2", BlackBoxRHS(base_network_f)),
]

_VARIABLE_ORDER_CLAIMS: list[Claim] = [
    _VariableOrderClimbClaim(),
    _VariableOrderDropClaim(),
]

_STIFFNESS_CLAIMS: list[Claim] = [
    _StiffnessDiagnosticClaim(),
    _FamilySwitchRoundTripClaim(),
    _StiffeningNetworkSwitchClaim(),
]

_VODE_CONTROLLER_CLAIMS: list[Claim] = [
    _VODEControllerSwitchClaim(),
]

_EXPONENTIAL_CLAIMS: list[Claim] = [
    _PhiFunctionClaim(),
]

_AUTO_DISPATCH_CLAIMS: list[_AutoDispatchClaim] = [
    _AutoDispatchClaim(
        label="explicit",
        rhs=BlackBoxRHS(base_network_f),
        state=ODEState(0.0, Tensor([1.0, 0.0])),
        dt=0.05,
        direct=rk4,
        expected_order=4,
    ),
    _AutoDispatchClaim(
        label="implicit",
        rhs=DECAY_RHS,
        state=ODEState(0.0, Tensor([1.0, 0.0, 0.0])),
        dt=0.05,
        direct=implicit_midpoint,
        expected_order=2,
    ),
    _AutoDispatchClaim(
        label="split",
        rhs=DECAY_RHS_IMEX,
        state=ODEState(0.0, Tensor([1.0, 0.0, 0.0])),
        dt=0.05,
        direct=ars222,
        expected_order=2,
    ),
    _AutoDispatchClaim(
        label="semilinear",
        rhs=etd_split_rhs(),
        state=ODEState(0.0, Tensor([1.0, 0.0, 0.0])),
        dt=0.05,
        direct=cox_matthews_etdrk4,
        expected_order=4,
    ),
    _AutoDispatchClaim(
        label="composite",
        rhs=split_rhs(),
        state=ODEState(0.0, Tensor([1.0, 0.0])),
        dt=0.05,
        direct=CompositionIntegrator([rk4, rk4], strang_steps(), order=2),
        expected_order=2,
    ),
    _AutoDispatchClaim(
        label="symplectic",
        rhs=HamiltonianRHS(dT_dp=lambda p: p, dV_dq=lambda q: q, split_index=1),
        state=ODEState(0.0, Tensor([1.0, 0.0])),
        dt=0.05,
        direct=leapfrog,
        expected_order=2,
    ),
]

_TYPE_COHERENCE_CLAIMS: list[_TypeCoherenceClaim] = [
    _TypeCoherenceClaim(
        ImplicitRungeKuttaIntegrator,
        [backward_euler, implicit_midpoint, crouzeix_3],
        "ImplicitRungeKuttaIntegrator",
    ),
    _TypeCoherenceClaim(
        AdditiveRungeKuttaIntegrator, [ars222], "AdditiveRungeKuttaIntegrator"
    ),
    _TypeCoherenceClaim(
        ExplicitMultistepIntegrator, [ab2, ab3, ab4], "ExplicitMultistepIntegrator"
    ),
    _TypeCoherenceClaim(
        SymplecticCompositionIntegrator,
        [symplectic_euler, leapfrog, forest_ruth, yoshida_6, yoshida_8],
        "SymplecticCompositionIntegrator",
    ),
    _TypeCoherenceClaim(
        CompositionIntegrator,
        [CompositionIntegrator([rk4, rk4], strang_steps(), order=2)],
        "CompositionIntegrator",
    ),
]

_NEWTON_SOLVE_CLAIMS: list[_NewtonSolveClaim] = [
    _NewtonSolveClaim(
        label="nonlinear_scalar",
        y_exp_vals=[0.9],
        gamma_dt=0.1,
        f_vals=lambda v: [v[0] * v[0]],
        jac_vals=lambda v: [[2.0 * v[0]]],
        y_star=[1.0],
    ),
    _NewtonSolveClaim(
        label="linear_system",
        y_exp_vals=[1.0, 2.0],
        gamma_dt=0.25,
        f_vals=lambda v: [v[0], 2.0 * v[1]],
        jac_vals=lambda v: [[1.0, 0.0], [0.0, 2.0]],
        y_star=[4.0 / 3.0, 4.0],
    ),
]

_NETWORK_INVARIANT_CLAIMS: list[_NetworkInvariantsClaim] = [
    _NetworkInvariantsClaim(),
]

_PROJECT_CONSERVED_CLAIMS: list[_ProjectConservedClaim] = [
    _ProjectConservedClaim(),
]

_PROJECTED_NEWTON_CLAIMS: list[_ProjectedNewtonClaim] = [
    _ProjectedNewtonClaim("decay_k1_gh01", k=1.0, gamma_dt=0.1, y_exp=[0.9, 0.1]),
    _ProjectedNewtonClaim("decay_k5_gh02", k=5.0, gamma_dt=0.2, y_exp=[0.7, 0.3]),
]

_CONSTRAINT_LIFECYCLE_CLAIMS: list[Claim] = [
    _ActivationClaim(),
    _NoChatteringClaim(),
    _ConsistentInitClaim(),
    _DeactivationClaim(),
]

_NSE_SOLVER_CLAIMS: list[Claim] = [
    _NSEDirectSolveClaim(),
    _NonlinearSolveScalarClaim(),
]

_RATE_THRESHOLD_CLAIMS: list[Claim] = [
    _RateThresholdClaim(),
]

_PARAMETRIC_NETWORK_CLAIMS: list[_ParametricNSEClaim] = [
    *chain_claims(_CHAIN_N),
    *spoke_claims(_SPOKE_N, _SPOKE_K),
]

_OFFLINE_NETWORK_CLAIMS: list[_ParametricNSEClaim] = [
    *chain_claims(_CHAIN_N_OFFLINE),
    *spoke_claims(_SPOKE_N_OFFLINE, _SPOKE_K_OFFLINE),
]

# Cost model: calibrate once at module load; build claims for all CI-range specs.
_COST_CALIBRATION: _CostCalibration = calibrate_cost()
_COST_MODEL_CLAIMS: list[_CostModelClaim] = [
    _CostModelClaim(spec._spec, _COST_CALIBRATION)  # type: ignore[attr-defined]
    for spec in _PARAMETRIC_NETWORK_CLAIMS
]

# ---------------------------------------------------------------------------
# Parametric test functions — one per claim family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim", _ORDER_CLAIMS, ids=[c.description for c in _ORDER_CLAIMS]
)
def test_rk_order_conditions(claim: _RKOrderClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _CONVERGENCE_CLAIMS, ids=[c.description for c in _CONVERGENCE_CLAIMS]
)
def test_convergence(claim: _ConvergenceClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _CONSERVATION_CLAIMS, ids=[c.description for c in _CONSERVATION_CLAIMS]
)
def test_conservation(claim: _ConservationClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _STEPPER_CLAIMS, ids=[c.description for c in _STEPPER_CLAIMS]
)
def test_stepper_advance(claim: _StepperClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _PI_ACCURACY_CLAIMS, ids=[c.description for c in _PI_ACCURACY_CLAIMS]
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
    "claim", _A_STABILITY_CLAIMS, ids=[c.description for c in _A_STABILITY_CLAIMS]
)
def test_a_stability(claim: _AStabilityClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _L_STABILITY_CLAIMS, ids=[c.description for c in _L_STABILITY_CLAIMS]
)
def test_l_stability(claim: _LStabilityClaim) -> None:
    claim.check()


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
def test_nordsieck_rescaled_accuracy(claim: _NordsieckRescaledAccuracyClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _VARIABLE_ORDER_CLAIMS, ids=[c.description for c in _VARIABLE_ORDER_CLAIMS]
)
def test_variable_order_nordsieck(claim: Claim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _STIFFNESS_CLAIMS, ids=[c.description for c in _STIFFNESS_CLAIMS]
)
def test_stiffness_switching(claim: Claim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _VODE_CONTROLLER_CLAIMS,
    ids=[c.description for c in _VODE_CONTROLLER_CLAIMS],
)
def test_vode_controller(claim: Claim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _EXPONENTIAL_CLAIMS, ids=[c.description for c in _EXPONENTIAL_CLAIMS]
)
def test_exponential_integrators(claim: Claim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _AUTO_DISPATCH_CLAIMS, ids=[c.description for c in _AUTO_DISPATCH_CLAIMS]
)
def test_auto_dispatch(claim: _AutoDispatchClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _TYPE_COHERENCE_CLAIMS, ids=[c.description for c in _TYPE_COHERENCE_CLAIMS]
)
def test_type_coherence(claim: _TypeCoherenceClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _NEWTON_SOLVE_CLAIMS, ids=[c.description for c in _NEWTON_SOLVE_CLAIMS]
)
def test_newton_solve(claim: _NewtonSolveClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _NETWORK_INVARIANT_CLAIMS,
    ids=[c.description for c in _NETWORK_INVARIANT_CLAIMS],
)
def test_reaction_network_invariants(claim: _NetworkInvariantsClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _PROJECT_CONSERVED_CLAIMS,
    ids=[c.description for c in _PROJECT_CONSERVED_CLAIMS],
)
def test_project_conserved(claim: _ProjectConservedClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _PROJECTED_NEWTON_CLAIMS,
    ids=[c.description for c in _PROJECTED_NEWTON_CLAIMS],
)
def test_projected_newton(claim: _ProjectedNewtonClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _CONSTRAINT_LIFECYCLE_CLAIMS,
    ids=[c.description for c in _CONSTRAINT_LIFECYCLE_CLAIMS],
)
def test_constraint_lifecycle(claim: Claim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _NSE_SOLVER_CLAIMS, ids=[c.description for c in _NSE_SOLVER_CLAIMS]
)
def test_nse_solver(claim: Claim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _RATE_THRESHOLD_CLAIMS, ids=[c.description for c in _RATE_THRESHOLD_CLAIMS]
)
def test_rate_threshold(claim: Claim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim",
    _PARAMETRIC_NETWORK_CLAIMS,
    ids=[c.description for c in _PARAMETRIC_NETWORK_CLAIMS],
)
def test_parametric_network(claim: _ParametricNSEClaim) -> None:
    claim.check()


@pytest.mark.parametrize(
    "claim", _COST_MODEL_CLAIMS, ids=[c.description for c in _COST_MODEL_CLAIMS]
)
def test_cost_model(claim: _CostModelClaim) -> None:
    claim.check()


_OFFLINE_SKIP_REASON = (
    "offline network stress tests; "
    "set COSMIC_FOUNDRY_OFFLINE_NETWORK_STRESS=1 to run"
)


@pytest.mark.parametrize(
    "claim",
    _OFFLINE_NETWORK_CLAIMS,
    ids=[c.description for c in _OFFLINE_NETWORK_CLAIMS],
)
@pytest.mark.skipif(not _OFFLINE_NETWORK_STRESS, reason=_OFFLINE_SKIP_REASON)
@pytest.mark.offline
def test_offline_network(claim: _ParametricNSEClaim) -> None:
    claim.check()
