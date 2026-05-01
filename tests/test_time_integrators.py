"""Time-integrator verification — outer-product parametric test suite.

Three test axes:
  test_convergence : _ORDERS × _PROBS — AutoIntegrator dispatches by RHS type; skips
                     unsupported (order, problem) pairs via ValueError
  test_correctness : _CORRECT_CLAIMS  — integration histories match analytical f(t)
  test_performance : _PERF_CLAIMS     — self-calibrated cost-to-accuracy rooflines
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.backends import (
    NumpyBackend,
    get_default_backend,
    set_default_backend,
)
from cosmic_foundry.computation.tensor import Tensor, norm
from tests.claims import CLAIM_WALLTIME_BUDGET_S, INTEGRATOR_CLAIM_BUDGET_S, Claim

_PREV = get_default_backend()
set_default_backend(NumpyBackend())
_BUDGET = INTEGRATOR_CLAIM_BUDGET_S
_U2, _U3 = Tensor([1.0, 0.0]), Tensor([1.0, 0.0, 0.0])
_PERF_TRIALS = 10
_PERF_BATCH = 200
_PERF_OVERHEAD = 80.0


@pytest.fixture(scope="module", autouse=True)
def _numpy_backend() -> Any:
    yield
    set_default_backend(_PREV)


# ── integration runner ────────────────────────────────────────────────────────


def _run(
    inst: Any, rhs: Any, u0: Tensor, dt: float, t_end: float = 1.0
) -> _ti.ODEState:  # noqa: E501
    """Integrate from t=0 to t≈t_end; bootstrap multistep methods."""
    n = round(t_end / dt)
    if hasattr(inst, "init_state"):
        state = inst.init_state(rhs, 0.0, u0, dt)
        for _ in range(max(n - inst.order, 1)):
            state = inst.step(rhs, state, dt)
    else:
        state = _ti.ODEState(0.0, u0)
        for _ in range(n):
            state = inst.step(rhs, state, dt)
    return state


def _history(inst: Any, rhs: Any, u0: Tensor, dt: float, t_end: float) -> list:
    state = _ti.ODEState(0.0, u0)
    states = []
    while state.t < t_end:
        state = inst.step(rhs, state, min(dt, t_end - state.t))
        states.append(state)
    return states


def _best_per_call(fn: Any, *, batch: int = _PERF_BATCH) -> float:
    best = float("inf")
    for _ in range(_PERF_TRIALS):
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best / batch


def _best_elapsed(fn: Any) -> tuple[float, Any]:
    best_elapsed = float("inf")
    best_result = None
    for _ in range(_PERF_TRIALS):
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        if elapsed < best_elapsed:
            best_elapsed = elapsed
            best_result = result
    return best_elapsed, best_result


# ── convergence slope (log-log OLS) ──────────────────────────────────────────


def _slope(errs: list[float], dts: list[float]) -> float:
    pts = [
        (math.log(d), math.log(e)) for d, e in zip(dts, errs, strict=False) if e > 1e-13
    ]
    if len(pts) < 3:
        pytest.skip(f"machine precision reached ({len(pts)} valid points)")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - mx) ** 2 for x in xs
    )


# ── exact solutions + error helpers ──────────────────────────────────────────


def _exact2(t: float) -> tuple[float, ...]:
    e = math.exp(-t)
    return e, 1.0 - e


def _exact3(t: float) -> tuple[float, ...]:
    e1, e2 = math.exp(-t), math.exp(-2.0 * t)
    return e1, e1 - e2, 1.0 - 2.0 * e1 + e2


def _exact_osc(t: float) -> tuple[float, ...]:
    return math.cos(t), math.sin(t)  # comp-split: u=(q,p) with u_dot=(-p,q)


def _exact_ham(t: float) -> tuple[float, ...]:
    return math.cos(t), -math.sin(t)  # Hamiltonian H=(q²+p²)/2: q=cos t, p=-sin t


def _exact_scalar_decay(t: float) -> tuple[float, ...]:
    return (math.exp(-t),)


def _exact_semilinear(t: float) -> tuple[float, ...]:
    forced = (
        math.exp(-2.0 * t)
        * (math.exp(2.0 * t) * (2.0 * math.sin(t) - math.cos(t)) + 1.0)
        / 5.0
    )
    return (math.exp(-2.0 * t) + forced,)


def _err(u: Tensor, exact: Any, t: float) -> float:
    return max(abs(float(u[i]) - v) for i, v in enumerate(exact(t)))


def _conserved(u: Tensor, n: int) -> bool:
    return abs(sum(float(u[i]) for i in range(n)) - 1.0) < 1e-10


def _scalar_decay_jacobian_rhs() -> _ti.JacobianRHS:
    return _ti.JacobianRHS(
        lambda t, u: Tensor([-float(u[0])], backend=u.backend),
        lambda t, u: Tensor([[-1.0]], backend=u.backend),
    )


# ── problem registry ──────────────────────────────────────────────────────────
# Each problem supplies ONE canonical RHS; AutoIntegrator dispatches by its type.
# (id, u0, n_species, exact_fn, mass_conserved, rhs)


def _build_probs() -> list:
    def f2(t, u):  # type: ignore[misc]
        return Tensor([-float(u[0]), float(u[0])], backend=u.backend)

    def f3(t, u):  # type: ignore[misc]
        x0, x1 = float(u[0]), float(u[1])
        return Tensor([-x0, x0 - 2.0 * x1, 2.0 * x1], backend=u.backend)

    def fA(t, u):  # type: ignore[misc]
        return Tensor([-float(u[1]), 0.0], backend=u.backend)

    def fB(t, u):  # type: ignore[misc]
        return Tensor([0.0, float(u[0])], backend=u.backend)

    def split_explicit(t, u):  # type: ignore[misc]
        return Tensor([-0.2 * float(u[0])], backend=u.backend)

    def split_implicit(t, u):  # type: ignore[misc]
        return Tensor([-0.8 * float(u[0])], backend=u.backend)

    def split_jacobian(t, u):  # type: ignore[misc]
        return Tensor([[-0.8]], backend=u.backend)

    def semilinear_forcing(t, u):  # type: ignore[misc]
        return Tensor([math.sin(t)], backend=u.backend)

    return [
        ("base2", _U2, 2, _exact2, True, _ti.BlackBoxRHS(f2)),
        ("decay3", _U3, 3, _exact3, True, _ti.BlackBoxRHS(f3)),
        (
            "jac_decay1",
            Tensor([1.0]),
            1,
            _exact_scalar_decay,
            False,
            _scalar_decay_jacobian_rhs(),
        ),
        (
            "split_decay1",
            Tensor([1.0]),
            1,
            _exact_scalar_decay,
            False,
            _ti.SplitRHS(split_explicit, split_implicit, split_jacobian),
        ),
        (
            "semilinear1",
            Tensor([1.0]),
            1,
            _exact_semilinear,
            False,
            _ti.SemilinearRHS(Tensor([[-2.0]]), semilinear_forcing),
        ),
        (
            "osc2",
            _U2,
            2,
            _exact_osc,
            False,
            _ti.CompositeRHS([_ti.BlackBoxRHS(fA), _ti.BlackBoxRHS(fB)]),
        ),
        (
            "ham2",
            _U2,
            2,
            _exact_ham,
            False,
            _ti.HamiltonianRHS(dT_dp=lambda p: p, dV_dq=lambda q: q, split_index=1),
        ),
    ]


_ORDERS = [1, 2, 3, 4, 5, 6]
_PROBS = _build_probs()


# ── Claim wrappers ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CorrectnessSpec:
    name: str
    run: Any
    expected: Any
    tol: float
    expected_walltime_s: float = 1.0
    xfail_reason: str | None = None


@dataclass(frozen=True)
class _IntegratorCalibration:
    scalar_rhs_s: float
    semilinear_nonlinear_s: float
    exp_action_s: float


@dataclass(frozen=True)
class _PerformanceSpec:
    name: str
    run: Any
    expected: Any
    tol: float
    roofline: Any


class _ConvergenceClaim(Claim[None]):
    """Convergence + conservation claim for one (order, problem) pair."""

    def __init__(self, order: int, prob: tuple) -> None:
        self._order = order
        self._prob = prob

    @property
    def description(self) -> str:
        return f"convergence/order{self._order}/{self._prob[0]}"

    def check(self, _calibration: None) -> None:
        pid, u0, n, exact, mass_cons, rhs = self._prob
        inst = _ti.AutoIntegrator(self._order)
        t0 = time.perf_counter()
        dts: list[float] = []
        errs: list[float] = []
        dt = 0.1
        state: _ti.ODEState | None = None
        while len(dts) < 8:
            try:
                state = _run(inst, rhs, u0, dt)
            except ValueError as e:
                pytest.skip(str(e))
            errs.append(_err(state.u, exact, state.t))
            dts.append(dt)
            dt /= 2.0
            if errs[-1] == 0.0:
                break
            if time.perf_counter() - t0 > _BUDGET:
                break
        assert (
            _slope(errs, dts) >= self._order - 0.3
        ), f"order{self._order}/{pid}: slope too low"
        if mass_cons and state is not None:
            assert _conserved(
                state.u, n
            ), f"order{self._order}/{pid}: mass not conserved"
            for i in range(n):
                assert (
                    float(state.u[i]) >= -1e-10
                ), f"order{self._order}/{pid}: u[{i}] negative"


class _CorrectnessClaim(Claim[None]):
    """Accuracy claim for one numerical history against an analytical f(t)."""

    def __init__(self, spec: _CorrectnessSpec) -> None:
        self._spec = spec

    @property
    def description(self) -> str:
        return f"correctness/{self._spec.name}"

    def check(self, _calibration: None) -> None:
        if self._spec.expected_walltime_s > CLAIM_WALLTIME_BUDGET_S:
            pytest.skip(
                f"{self.description}: expected {self._spec.expected_walltime_s:.1f}s "
                f"> walltime budget {CLAIM_WALLTIME_BUDGET_S:.1f}s"
            )
        if self._spec.xfail_reason is not None:
            try:
                self._check_history()
            except AssertionError:
                pytest.xfail(self._spec.xfail_reason)
            raise AssertionError(f"{self.description}: expected failure passed")
        self._check_history()

    def _check_history(self) -> None:
        for state in self._spec.run():
            assert _err(state.u, self._spec.expected, state.t) < self._spec.tol


class _PerformanceClaim(Claim[_IntegratorCalibration]):
    """Cost-to-accuracy claim against locally measured primitive rooflines."""

    def __init__(self, spec: _PerformanceSpec) -> None:
        self._spec = spec

    @property
    def description(self) -> str:
        return f"performance/{self._spec.name}"

    def check(self, calibration: _IntegratorCalibration) -> None:
        elapsed, state = _best_elapsed(self._spec.run)
        err = _err(state.u, self._spec.expected, state.t)
        assert err < self._spec.tol
        roofline = self._spec.roofline(calibration)
        assert elapsed <= _PERF_OVERHEAD * roofline, (
            f"{self.description}: {elapsed:.3e}s actual, "
            f"{roofline:.3e}s calibrated roofline, "
            f"{elapsed / roofline:.1f}x > {_PERF_OVERHEAD:.1f}x"
        )


# ── parametric network spec + NSE helpers ─────────────────────────────────────


@dataclass
class _Spec:
    name: str
    topo: str
    n: int
    rates: list[float]

    @property
    def p(self) -> int:
        return self.n - 1

    def t_end(self) -> float:
        k = min(self.rates)
        return 4.0 * self.p**2 / k if self.topo == "chain" else 8.0 / k

    def dt0(self) -> float:
        return min(0.05, 0.1 / max(self.rates))

    def u0(self) -> Tensor:
        return Tensor([1.0] + [0.0] * self.p)

    def build_rhs(self) -> _ti.ReactionNetworkRHS:
        n, p, rates, topo = self.n, self.p, list(self.rates), self.topo
        rows = [[0.0] * p for _ in range(n)]
        for j in range(p):
            rows[j if topo == "chain" else 0][j] = -1.0
            rows[j + 1][j] = 1.0
        S = Tensor(rows)

        def rp(t: float, u: Tensor) -> Tensor:
            idx = lambda j: j if topo == "chain" else 0  # noqa: E731
            return Tensor(
                [rates[j] * float(u[idx(j)]) for j in range(p)], backend=u.backend
            )

        def rm(t: float, u: Tensor) -> Tensor:
            return Tensor(
                [rates[j] * float(u[j + 1]) for j in range(p)], backend=u.backend
            )

        return _ti.ReactionNetworkRHS(S, rp, rm, self.u0())


def _chain_specs(nr: range, k: float = 1.0) -> list[_Spec]:
    return [_Spec(f"chain-n{n}-k{k:.0f}", "chain", n, [k] * (n - 1)) for n in nr]


def _spoke_specs(nr: range, ks: list[int]) -> list[_Spec]:
    specs = []
    for n in nr:
        p, nf = n - 1, (n - 1) // 2
        for k in ks:
            specs.append(
                _Spec(
                    f"spoke-n{n}-k{k:.0f}",
                    "spoke",
                    n,
                    [float(k)] * nf + [1.0] * (p - nf),
                )
            )
    return specs


_CI_SPECS = _chain_specs(range(3, 5)) + _spoke_specs(range(3, 7), [1, 10])


def _ode_correctness_specs() -> list[_CorrectnessSpec]:
    specs = []
    for name, u0, _n, exact, _mass, rhs in _PROBS:
        specs.append(
            _CorrectnessSpec(
                name=name,
                run=lambda u0=u0, rhs=rhs: _history(
                    _ti.AutoIntegrator(4), rhs, u0, 0.02, 0.4
                ),
                expected=exact,
                tol=1e-5,
            )
        )
    return specs


def _nse_correctness_spec(
    spec: _Spec, *, expected_walltime_s: float = 1.0
) -> _CorrectnessSpec:
    def run() -> list[_ti.ODEState]:
        rhs = spec.build_rhs()
        ctrl = _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=_ti.ImplicitRungeKuttaIntegrator(2),
            inner=_ti.PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=spec.dt0()),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        return [ctrl.advance(spec.u0(), 0.0, spec.t_end())]

    def expected(t: float) -> tuple[float, ...]:
        return (1.0 / spec.n,) * spec.n

    return _CorrectnessSpec(
        f"nse/{spec.name}", run, expected, 1e-7, expected_walltime_s
    )


def _nse_transient_correctness_spec() -> _CorrectnessSpec:
    k = 5.0

    def run() -> list[_ti.ODEState]:
        rhs = _ti.ReactionNetworkRHS(
            Tensor([[-1.0], [1.0]]),
            lambda t, u: Tensor([k * float(u[0])]),
            lambda t, u: Tensor([k * float(u[1])]),
            Tensor([0.9, 0.1]),
        )
        ctrl = _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=_ti.ImplicitRungeKuttaIntegrator(2),
            inner=_ti.PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=0.01),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        return [ctrl.advance(Tensor([0.7, 0.3]), 0.0, 0.1)]

    def expected(t: float) -> tuple[float, float]:
        x0 = 0.5 + 0.2 * math.exp(-2.0 * k * t)
        return x0, 1.0 - x0

    return _CorrectnessSpec("nse/transient", run, expected, 1e-4)


RateFn = Callable[[float], list[tuple[int, int, float]]]


def _linear_network_rhs(rate_fn: RateFn) -> _ti.JacobianRHS:
    """Build a mass-conserving linear reaction network RHS from edge rates."""

    def f(t: float, u: Tensor) -> Tensor:
        n = u.shape[0]
        out = [0.0] * n
        for src, dst, rate in rate_fn(t):
            flux = rate * float(u[src])
            out[src] -= flux
            out[dst] += flux
        return Tensor(out, backend=u.backend)

    def jac(t: float, u: Tensor) -> Tensor:
        n = u.shape[0]
        mat = [[0.0 for _ in range(n)] for _ in range(n)]
        for src, dst, rate in rate_fn(t):
            mat[src][src] -= rate
            mat[dst][src] += rate
        return Tensor(mat, backend=u.backend)

    return _ti.JacobianRHS(f=f, jac=jac)


def _alpha_chain_rates(n: int) -> RateFn:
    rates = [10.0 ** (-2.0 + 8.0 * i / (n - 2)) for i in range(n - 1)]

    def edges(t: float) -> list[tuple[int, int, float]]:
        return [(i, i + 1, rates[i]) for i in range(n - 1)]

    return edges


def _branched_hot_window_rates(n: int) -> RateFn:
    base_edges = [(i, i + 1, 0.04 * (1.25**i)) for i in range(n - 1)]
    branches = [(2, 7, 0.03), (4, 10, 0.02), (6, 13, 0.015), (8, 15, 0.012)]

    def edges(t: float) -> list[tuple[int, int, float]]:
        hot = 1.0 + 2.0e4 * math.exp(-(((t - 0.18) / 0.025) ** 2))
        breakout = 1.0 + 5.0e5 * math.exp(-(((t - 0.24) / 0.018) ** 2))
        result = [(src, dst, rate * hot) for src, dst, rate in base_edges]
        result.extend((src, dst, rate * breakout) for src, dst, rate in branches)
        return result

    return edges


def _vode_controller(*, q_max: int = 6) -> _ti.VODEController:
    return _ti.VODEController(
        order_selector=_ti.OrderSelector(
            q_min=2,
            q_max=q_max,
            atol=2e-5,
            rtol=2e-5,
            factor_min=0.2,
            factor_max=1.15,
        ),
        stiffness_switcher=_ti.StiffnessSwitcher(
            stiff_threshold=1.0,
            nonstiff_threshold=0.35,
        ),
        q_initial=2,
        initial_family="adams",
        max_rejections=80,
    )


def _assert_abundance_state(u: Tensor, *, label: str) -> None:
    total = sum(float(u[i]) for i in range(u.shape[0]))
    assert abs(total - 1.0) < 1e-10, f"{label}: mass drift {total - 1.0:.3e}"
    minimum = min(float(u[i]) for i in range(u.shape[0]))
    assert minimum >= -1e-8, f"{label}: minimum abundance {minimum:.3e}"


def _alpha_chain_stress_spec() -> _CorrectnessSpec:
    def run() -> list[_ti.ODEState]:
        rhs = _linear_network_rhs(_alpha_chain_rates(13))
        controller = _vode_controller()
        state = controller.advance(
            rhs,
            Tensor([1.0] + [0.0] * 12),
            t0=0.0,
            t_end=0.04,
            dt0=2e-4,
        )
        _assert_abundance_state(state.u, label="alpha_chain")
        assert controller.family_switches >= 1
        assert "bdf" in controller.accepted_families
        assert max(controller.accepted_stiffness) > 1.0
        assert controller.rejected_steps < 80
        return [state]

    return _CorrectnessSpec(
        "stress/vode_alpha_chain_rate_contrast",
        run,
        lambda t: (1.0,) * 13,
        float("inf"),
        expected_walltime_s=20.0,
    )


def _branched_hot_window_stress_spec() -> _CorrectnessSpec:
    def run() -> list[_ti.ODEState]:
        rhs = _linear_network_rhs(_branched_hot_window_rates(16))
        coarse = _vode_controller()
        fine = _vode_controller()
        u0 = Tensor([1.0] + [0.0] * 15)
        coarse_state = coarse.advance(rhs, u0, t0=0.0, t_end=0.32, dt0=5e-4)
        fine_state = fine.advance(rhs, u0, t0=0.0, t_end=0.32, dt0=2.5e-4)
        _assert_abundance_state(coarse_state.u, label="branched_hot_window/coarse")
        _assert_abundance_state(fine_state.u, label="branched_hot_window/fine")
        assert "adams" in coarse.accepted_families
        assert "bdf" in coarse.accepted_families
        assert coarse.family_switches >= 1
        assert max(coarse.accepted_stiffness) > 1.0
        assert float(norm(coarse_state.u - fine_state.u)) < 5e-2
        return [coarse_state]

    return _CorrectnessSpec(
        "stress/vode_branched_hot_window_self_consistency",
        run,
        lambda t: (1.0,) * 16,
        float("inf"),
        expected_walltime_s=30.0,
        xfail_reason=(
            "current VODE controller does not enforce positivity; this branched "
            "hot-window network exposes small negative abundances"
        ),
    )


def _explicit_rk4_performance_spec() -> _PerformanceSpec:
    rhs = _ti.BlackBoxRHS(lambda t, u: Tensor([-float(u[0])], backend=u.backend))
    dt, t_end, order = 0.02, 1.0, 4
    n_steps = round(t_end / dt)

    def run() -> _ti.ODEState:
        return _run(_ti.RungeKuttaIntegrator(order), rhs, Tensor([1.0]), dt, t_end)

    def roofline(cal: _IntegratorCalibration) -> float:
        return n_steps * order * cal.scalar_rhs_s

    return _PerformanceSpec(
        "explicit_rk4/scalar_decay",
        run,
        _exact_scalar_decay,
        1e-7,
        roofline,
    )


def _semilinear_lawson4_performance_spec() -> _PerformanceSpec:
    linear = Tensor([[-2.0]])

    def nonlinear(t: float, u: Tensor) -> Tensor:
        return Tensor([math.sin(t)], backend=u.backend)

    rhs = _ti.SemilinearRHS(linear, nonlinear)
    dt, t_end, order = 0.02, 0.4, 4
    n_steps = round(t_end / dt)

    def run() -> _ti.ODEState:
        return _run(
            _ti.LawsonRungeKuttaIntegrator(order), rhs, Tensor([1.0]), dt, t_end
        )

    def roofline(cal: _IntegratorCalibration) -> float:
        # Lawson RK4 stages require nonlinear evaluations plus dense linear-flow
        # actions; both primitives are calibrated on this device before testing.
        return n_steps * (4 * cal.semilinear_nonlinear_s + 12 * cal.exp_action_s)

    return _PerformanceSpec(
        "lawson_rk4/semilinear_forcing",
        run,
        _exact_semilinear,
        1e-7,
        roofline,
    )


@pytest.fixture(scope="module")
def integrator_calibration() -> _IntegratorCalibration:
    u = Tensor([1.0])

    def scalar_rhs() -> Tensor:
        return Tensor([-float(u[0])], backend=u.backend)

    def nonlinear() -> Tensor:
        return Tensor([math.sin(0.2)], backend=u.backend)

    def exp_action() -> Tensor:
        return _ti.PhiFunction(0).apply(Tensor([[-0.04]]), u)

    return _IntegratorCalibration(
        scalar_rhs_s=_best_per_call(scalar_rhs),
        semilinear_nonlinear_s=_best_per_call(nonlinear),
        exp_action_s=_best_per_call(exp_action),
    )


# ── claim registries ─────────────────────────────────────────────────────────

_CONV_CLAIMS: list[Claim[None]] = [
    _ConvergenceClaim(order, prob) for order in _ORDERS for prob in _PROBS
]

_OFF_SPECS = _chain_specs(range(5, 12)) + _spoke_specs(range(7, 22), [1, 10, 100])
_CORRECT_CLAIMS: list[Claim[None]] = [
    *[_CorrectnessClaim(s) for s in _ode_correctness_specs()],
    *[_CorrectnessClaim(_nse_correctness_spec(s)) for s in _CI_SPECS],
    _CorrectnessClaim(_nse_transient_correctness_spec()),
    *[
        _CorrectnessClaim(_nse_correctness_spec(s, expected_walltime_s=5.0))
        for s in _OFF_SPECS
    ],
    _CorrectnessClaim(_alpha_chain_stress_spec()),
    _CorrectnessClaim(_branched_hot_window_stress_spec()),
]
_PERF_CLAIMS: list[Claim[_IntegratorCalibration]] = [
    _PerformanceClaim(_explicit_rk4_performance_spec()),
    _PerformanceClaim(_semilinear_lawson4_performance_spec()),
]


# ── parametric test functions (each body is a single claim.check() dispatch) ──


_CONV_IDS = [c.description for c in _CONV_CLAIMS]


@pytest.mark.parametrize("claim", _CONV_CLAIMS, ids=_CONV_IDS)
def test_convergence(claim: Claim[None]) -> None:
    claim.check(None)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[None]) -> None:
    claim.check(None)


@pytest.mark.parametrize(
    "claim", _PERF_CLAIMS, ids=[c.description for c in _PERF_CLAIMS]
)
def test_performance(
    claim: Claim[_IntegratorCalibration],
    integrator_calibration: _IntegratorCalibration,
) -> None:
    claim.check(integrator_calibration)
