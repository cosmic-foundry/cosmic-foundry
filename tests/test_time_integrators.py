"""Time-integrator verification — outer-product parametric test suite.

Three test axes:
  test_convergence : _ORDERS × _PROBS — AutoIntegrator dispatches by RHS type; skips
                     unsupported (order, problem) pairs via ValueError
  test_nse         : _CI_SPECS        — NSE detection for parametric networks
  test_behavior    : _CHECKS          — targeted checks for non-grid behavior
"""

from __future__ import annotations

import math
import os
import time
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
from tests.claims import INTEGRATOR_CLAIM_BUDGET_S, Claim

_PREV = get_default_backend()
set_default_backend(NumpyBackend())
_BUDGET = INTEGRATOR_CLAIM_BUDGET_S
_OFFLINE = os.environ.get("COSMIC_FOUNDRY_OFFLINE_NETWORK_STRESS") == "1"
_U2, _U3 = Tensor([1.0, 0.0]), Tensor([1.0, 0.0, 0.0])


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


def _err(u: Tensor, exact: Any, t: float) -> float:
    return max(abs(float(u[i]) - v) for i, v in enumerate(exact(t)))


def _conserved(u: Tensor, n: int) -> bool:
    return abs(sum(float(u[i]) for i in range(n)) - 1.0) < 1e-10


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

    return [
        ("base2", _U2, 2, _exact2, True, _ti.BlackBoxRHS(f2)),
        ("decay3", _U3, 3, _exact3, True, _ti.BlackBoxRHS(f3)),
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


class _ConvergenceClaim(Claim):
    """Convergence + conservation claim for one (order, problem) pair."""

    def __init__(self, order: int, prob: tuple) -> None:
        self._order = order
        self._prob = prob

    @property
    def description(self) -> str:
        return f"convergence/order{self._order}/{self._prob[0]}"

    def check(self) -> None:
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


class _NSEClaim(Claim):
    """NSE-detection claim for one parametric reaction network."""

    def __init__(self, spec: _Spec) -> None:
        self._spec = spec

    @property
    def description(self) -> str:
        return f"nse/{self._spec.name}"

    def check(self) -> None:
        spec = self._spec
        rhs = spec.build_rhs()
        ctrl = _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=_ti.ImplicitRungeKuttaIntegrator(2),
            inner=_ti.PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=spec.dt0()),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        state = ctrl.advance(spec.u0(), 0.0, spec.t_end())
        assert ctrl.nse_events, f"no NSE events: {spec.name}"
        assert state.active_constraints == frozenset(range(spec.p))
        eq = 1.0 / spec.n
        for i in range(spec.n):
            assert abs(float(state.u[i]) - eq) < 1e-7, f"{spec.name}[{i}]"


class _BehaviorClaim(Claim):
    """Targeted behavior check for a specific integrator feature."""

    def __init__(self, fn: Any, id_: str) -> None:
        self._fn = fn
        self._id = id_

    @property
    def description(self) -> str:
        return f"behavior/{self._id}"

    def check(self) -> None:
        self._fn()


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


# ── targeted behavior check implementations ───────────────────────────────────


def _etd_check() -> None:
    """ETD4RK convergence against a Richardson-extrapolated reference."""
    _L = Tensor([[-8.0, 0.0, 0.0], [8.0, -0.5, 0.0], [0.0, 0.5, 0.0]])

    def _N(t: float, u: Tensor) -> Tensor:
        r = 0.25 + 0.1 * math.sin(2.0 * t)
        return Tensor([-r * float(u[0]), 0.0, r * float(u[0])], backend=u.backend)

    rhs = _ti.SemilinearRHS(_L, _N)
    t_end = 0.5
    ref = _run(_ti.CoxMatthewsETDRK4Integrator(4), rhs, _U3, t_end / 128, t_end)
    t0, dts, errs = time.perf_counter(), [], []
    for i in range(1, 7):
        if time.perf_counter() - t0 > _BUDGET:
            break
        dt = t_end / 2**i
        s = _run(_ti.CoxMatthewsETDRK4Integrator(4), rhs, _U3, dt, t_end)
        errs.append(float(norm(s.u - ref.u)))
        dts.append(dt)
    assert _slope(errs, dts) >= 3.5


def _semilinear_orders_check() -> None:
    """Lawson RK semilinear integrators converge at orders 1 through 6."""
    L = Tensor([[-2.0]])

    def _N(t: float, u: Tensor) -> Tensor:
        return Tensor([math.sin(t)], backend=u.backend)

    def _exact(t: float) -> tuple[float]:
        forced = (
            math.exp(-2.0 * t)
            * (math.exp(2.0 * t) * (2.0 * math.sin(t) - math.cos(t)) + 1.0)
            / 5.0
        )
        return (math.exp(-2.0 * t) + forced,)

    rhs = _ti.SemilinearRHS(L, _N)
    for q in range(1, 7):
        inst = _ti.LawsonRungeKuttaIntegrator(q)
        dts = [0.2, 0.1, 0.05, 0.025]
        errs = []
        for dt in dts:
            state = _run(inst, rhs, Tensor([1.0]), dt, t_end=1.0)
            errs.append(_err(state.u, _exact, state.t))
        assert _slope(errs, dts) >= q - 0.5, f"Lawson RK order {q}"

        auto_state = _run(_ti.AutoIntegrator(q), rhs, Tensor([1.0]), 0.025, t_end=1.0)
        direct_state = _run(inst, rhs, Tensor([1.0]), 0.025, t_end=1.0)
        assert float(norm(auto_state.u - direct_state.u)) < 1e-14


def _nordsieck_check() -> None:
    """NordsieckHistory change_order and rescale_step round-trip invariants."""
    nh = _ti.NordsieckHistory(
        h=0.1, z=(Tensor([0.8, 0.2]), Tensor([-0.08, 0.08]), Tensor([0.004, -0.004]))
    )
    raised = nh.change_order(4)
    lowered = raised.change_order(2)
    assert lowered.q == nh.q and float(norm(lowered.z[0] - nh.z[0])) == 0.0
    assert raised.q == 4 and float(norm(raised.z[3])) == 0.0
    rescaled = nh.rescale_step(0.05).rescale_step(0.1)
    assert rescaled.h == nh.h
    for a, b in zip(rescaled.z, nh.z, strict=True):
        assert float(norm(a - b)) < 1e-14


def _rk_order_conditions_check() -> None:
    """Explicit RK tableaux satisfy rooted-tree order conditions through p=6."""
    for q in range(1, 7):
        inst = _ti.RungeKuttaIntegrator(q)
        for tree in _ti.trees_up_to_order(q):
            assert _ti.elementary_weight(tree, inst.A_sym, inst.b_sym) == (
                1 / _ti.gamma(tree)
            ), f"RK{q}: failed tree {tree}"


def _scalar_decay_rhs() -> _ti.BlackBoxRHS:
    return _ti.BlackBoxRHS(lambda t, u: Tensor([-float(u[0])], backend=u.backend))


def _scalar_decay_jacobian_rhs() -> _ti.JacobianRHS:
    return _ti.JacobianRHS(
        lambda t, u: Tensor([-float(u[0])], backend=u.backend),
        lambda t, u: Tensor([[-1.0]], backend=u.backend),
    )


def _assert_scalar_decay_order(
    inst: Any, rhs: Any, q: int, *, tol: float = 0.5
) -> None:
    dts = [0.1, 0.05, 0.025, 0.0125]
    errs = []
    for dt in dts:
        state = _run(inst, rhs, Tensor([1.0]), dt, t_end=0.96)
        errs.append(abs(float(state.u[0]) - math.exp(-state.t)))
    assert _slope(errs, dts) >= q - tol, f"{type(inst).__name__} order {q}"


def _explicit_multistep_orders_check() -> None:
    """Adams-Bashforth fixed-order methods converge at orders 1 through 6."""
    rhs = _scalar_decay_rhs()
    for q in range(1, 7):
        _assert_scalar_decay_order(_ti.ExplicitMultistepIntegrator.for_order(q), rhs, q)


def _nordsieck_fixed_orders_check() -> None:
    """Nordsieck Adams and BDF fixed-order methods are instantiable through order 6."""
    plain = _scalar_decay_rhs()
    jac = _scalar_decay_jacobian_rhs()
    for q in range(1, 7):
        adams = _ti.MultistepIntegrator("adams", q)
        bdf = _ti.MultistepIntegrator("bdf", q)
        assert adams.order == q
        assert bdf.order == q

        adams_state = adams.init_state(plain, 0.0, Tensor([1.0]), 0.01)
        bdf_state = bdf.init_state(jac, 0.0, Tensor([1.0]), 0.01)
        assert adams.step(plain, adams_state, 0.01).history.q == q
        assert bdf.step(jac, bdf_state, 0.01).history.q == q


def _phi_check() -> None:
    """φ_k functions satisfy the correct Taylor recurrence for nilpotent A."""
    A, v = Tensor([[0.0, 1.0], [0.0, 0.0]]), Tensor([0.0, 1.0])
    for k, (a, b) in enumerate([(1.0, 1.0), (0.5, 1.0), (1 / 6, 0.5), (1 / 24, 1 / 6)]):
        assert float(norm(_ti.PhiFunction(k).apply(A, v) - Tensor([a, b]))) < 1e-14


def _variable_order_check() -> None:
    """VariableOrderNordsieckIntegrator climbs to q=4, drops to q=2 on sharpening."""
    sel = _ti.OrderSelector(q_min=2, q_max=4, atol=1e-4, rtol=1e-4)
    inst = _ti.VariableOrderNordsieckIntegrator("adams", sel)
    bb = _ti.BlackBoxRHS(
        lambda t, u: Tensor(
            [-float(u[0]), float(u[0]) - 2.0 * float(u[1]), 2.0 * float(u[1])],
            backend=u.backend,
        )
    )
    state = inst.advance(bb, _U3, t0=0.0, t_end=1.5, dt0=0.025)
    assert max(inst.accepted_orders) == 4 and inst.accepted_orders[-1] == 4
    assert _err(state.u, _exact3, state.t) < 5e-4

    def _sharp(t: float, u: Tensor) -> Tensor:
        k2 = 1.0 if t < 0.5 else 10.0
        x0, x1 = float(u[0]), float(u[1])
        return Tensor([-x0, x0 - k2 * x1, k2 * x1], backend=u.backend)

    inst2 = _ti.VariableOrderNordsieckIntegrator("adams", sel, q_initial=4)
    state2 = inst2.advance(_ti.BlackBoxRHS(_sharp), _U3, t0=0.0, t_end=1.0, dt0=0.02)
    post = [
        q
        for t, q in zip(inst2.accepted_times, inst2.accepted_orders, strict=True)
        if t > 0.55
    ]
    assert post and min(post) == 2
    assert _conserved(state2.u, 3)


def _vode_check() -> None:
    """VODEController switches from Adams to BDF on a fast/slow stiffening problem."""

    def _fv(t: float, u: Tensor) -> Tensor:
        k2 = 1.0 if t < 0.45 else 1000.0
        x0, x1 = float(u[0]), float(u[1])
        return Tensor([-x0, x0 - k2 * x1, k2 * x1], backend=u.backend)

    def _jv(t: float, u: Tensor) -> Tensor:
        k2 = 1.0 if t < 0.45 else 1000.0
        return Tensor([[-1.0, 0.0, 0.0], [1.0, -k2, 0.0], [0.0, k2, 0.0]])

    ctrl = _ti.VODEController(
        order_selector=_ti.OrderSelector(
            q_min=2, q_max=4, atol=5e-4, rtol=5e-4, factor_min=0.25, factor_max=1.15
        ),
        stiffness_switcher=_ti.StiffnessSwitcher(
            stiff_threshold=1.0, nonstiff_threshold=0.4
        ),
        q_initial=2,
        initial_family="adams",
    )
    ctrl.advance(_ti.JacobianRHS(_fv, _jv), _U3, t0=0.0, t_end=0.7, dt0=0.005)
    early = [
        f
        for t, f in zip(ctrl.accepted_times, ctrl.accepted_families, strict=True)
        if t < 0.35
    ]
    late = [
        f
        for t, f in zip(ctrl.accepted_times, ctrl.accepted_families, strict=True)
        if t > 0.5
    ]
    assert set(early) == {"adams"} and "bdf" in late
    assert max(ctrl.accepted_orders) > 2


def _lifecycle_check() -> None:
    """ConstraintAwareController: activation, no-chatter, consistent init, deactivation."""  # noqa: E501
    k = 5.0
    S = Tensor([[-1.0], [1.0]])
    rhs = _ti.ReactionNetworkRHS(
        S,
        lambda t, u: Tensor([k * float(u[0])]),
        lambda t, u: Tensor([k * float(u[1])]),
        Tensor([0.9, 0.1]),
    )

    def _ctrl() -> _ti.ConstraintAwareController:
        return _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=_ti.ImplicitRungeKuttaIntegrator(2),
            inner=_ti.PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=0.01),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )

    c = _ctrl()
    s = c.advance(Tensor([0.99, 0.01]), 0.0, 6.0)
    assert s.active_constraints == frozenset({0}) and c.activation_events
    assert not c.deactivation_events
    rp = float(rhs.forward_rate(s.t, s.u)[0])
    rm = float(rhs.reverse_rate(s.t, s.u)[0])
    assert abs(rp - rm) / max(abs(rp), abs(rm), 1e-100) < 0.01
    c2 = _ctrl()
    # integrate only 1 time-constant (0.1 s); system stays far from equilibrium
    # so the initial constraint deactivates and does not re-activate
    s2 = c2.advance(Tensor([0.7, 0.3]), 0.0, 0.1, initial_active=frozenset({0}))
    assert c2.deactivation_events and s2.active_constraints == frozenset()


def _nse_direct_check() -> None:
    """solve_nse finds the detailed-balance equilibrium X=(1/3, 1/3, 1/3)."""
    S = Tensor([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    rhs = _ti.ReactionNetworkRHS(
        S,
        lambda t, u: Tensor([2.0 * float(u[0]), 2.0 * float(u[1])]),
        lambda t, u: Tensor([2.0 * float(u[2]), 2.0 * float(u[2])]),
        Tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
    )
    result = _ti.solve_nse(rhs, Tensor([0.34, 0.33, 0.33]), t=0.0)
    for i in range(3):
        assert abs(float(result[i]) - 1.0 / 3.0) < 1e-10, f"species {i}"


# ── claim registries ─────────────────────────────────────────────────────────

_CONV_CLAIMS: list[Claim] = [
    _ConvergenceClaim(order, prob) for order in _ORDERS for prob in _PROBS
]

_NSE_CLAIMS: list[Claim] = [_NSEClaim(s) for s in _CI_SPECS]

_BEHAVIOR_CLAIMS: list[Claim] = [
    _BehaviorClaim(_rk_order_conditions_check, "rk/order_conditions_1_6"),
    _BehaviorClaim(_explicit_multistep_orders_check, "adams_bashforth/orders_1_6"),
    _BehaviorClaim(_nordsieck_fixed_orders_check, "nordsieck/fixed_orders_1_6"),
    _BehaviorClaim(_etd_check, "etd/convergence"),
    _BehaviorClaim(_semilinear_orders_check, "semilinear/orders_1_6"),
    _BehaviorClaim(_nordsieck_check, "nordsieck/round_trip"),
    _BehaviorClaim(_phi_check, "phi_function/coefficients"),
    _BehaviorClaim(_variable_order_check, "variable_order/climb_and_drop"),
    _BehaviorClaim(_vode_check, "vode/family_switch"),
    _BehaviorClaim(_lifecycle_check, "constraint/lifecycle"),
    _BehaviorClaim(_nse_direct_check, "nse/direct_solve"),
]

_OFF_REASON = (
    "offline network stress tests; "
    "set COSMIC_FOUNDRY_OFFLINE_NETWORK_STRESS=1 to run"
)
_OFF_SPECS = _chain_specs(range(5, 12)) + _spoke_specs(range(7, 22), [1, 10, 100])
_OFF_CLAIMS: list[Claim] = [_NSEClaim(s) for s in _OFF_SPECS]


# ── parametric test functions (each body is a single claim.check() dispatch) ──


_CONV_IDS = [c.description for c in _CONV_CLAIMS]


@pytest.mark.parametrize("claim", _CONV_CLAIMS, ids=_CONV_IDS)
def test_convergence(claim: Claim) -> None:
    claim.check()


@pytest.mark.parametrize("claim", _NSE_CLAIMS, ids=[c.description for c in _NSE_CLAIMS])
def test_nse(claim: Claim) -> None:
    claim.check()


_BEHAVIOR_IDS = [c.description for c in _BEHAVIOR_CLAIMS]


@pytest.mark.parametrize("claim", _BEHAVIOR_CLAIMS, ids=_BEHAVIOR_IDS)
def test_behavior(claim: Claim) -> None:
    claim.check()


@pytest.mark.parametrize("claim", _OFF_CLAIMS, ids=[c.description for c in _OFF_CLAIMS])
@pytest.mark.skipif(not _OFFLINE, reason=_OFF_REASON)
@pytest.mark.offline
def test_offline_nse(claim: Claim) -> None:
    claim.check()
