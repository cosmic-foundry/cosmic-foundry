"""Time-integrator verification — outer-product parametric test suite.

Two test axes:
  test_convergence : _ORDERS × _PROBS — AutoIntegrator dispatches by RHS type; skips
                     unsupported (order, problem) pairs via ValueError
  test_nse         : _NSE_CLAIMS      — integrated NSE behavior for reaction networks
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
from cosmic_foundry.computation.tensor import Tensor
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
    """NSE activation claim for one parametric reaction network."""

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


class _NSETransientClaim(Claim):
    """NSE constraints deactivate when a reaction network is far from equilibrium."""

    @property
    def description(self) -> str:
        return "nse/transient-non-equilibrium"

    def check(self) -> None:
        k = 5.0
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
        state = ctrl.advance(
            Tensor([0.7, 0.3]), 0.0, 0.1, initial_active=frozenset({0})
        )
        assert state.active_constraints == frozenset()
        assert abs(float(state.u[0]) + float(state.u[1]) - 1.0) < 1e-10
        assert abs(float(state.u[0]) - float(state.u[1])) > 0.05


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


# ── claim registries ─────────────────────────────────────────────────────────

_CONV_CLAIMS: list[Claim] = [
    _ConvergenceClaim(order, prob) for order in _ORDERS for prob in _PROBS
]

_NSE_CLAIMS: list[Claim] = [_NSEClaim(s) for s in _CI_SPECS] + [_NSETransientClaim()]

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


@pytest.mark.parametrize("claim", _OFF_CLAIMS, ids=[c.description for c in _OFF_CLAIMS])
@pytest.mark.skipif(not _OFFLINE, reason=_OFF_REASON)
@pytest.mark.offline
def test_offline_nse(claim: Claim) -> None:
    claim.check()
