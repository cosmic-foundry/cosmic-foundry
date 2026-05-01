"""Time-integrator verification — outer-product parametric test suite.

Three test axes:
  test_convergence : _ORDERS × _PROBS — AutoIntegrator dispatches by RHS type; skips
                     unsupported (order, problem) pairs via ValueError
  test_correctness : _CORRECT_CLAIMS  — integration histories match analytical f(t)
  test_performance : _PERF_CLAIMS     — execution-plan Tensor roofline checks
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
)
from cosmic_foundry.computation.tensor import Tensor, norm
from tests.claims import (
    BatchedFailure,
    Claim,
    ExecutionPlan,
)

_TIME_BACKEND = NumpyBackend()
_PERF_TRIALS = 10
_PERF_OVERHEAD = 80.0
_GPU_MIN_PERF_FMAS = 1.0e6


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


def _build_probs(backend: Any = _TIME_BACKEND) -> list:
    def tensor(data: Any) -> Tensor:
        return Tensor(data, backend=backend)

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
        ("base2", tensor([1.0, 0.0]), 2, _exact2, True, _ti.BlackBoxRHS(f2)),
        (
            "decay3",
            tensor([1.0, 0.0, 0.0]),
            3,
            _exact3,
            True,
            _ti.BlackBoxRHS(f3),
        ),
        (
            "jac_decay1",
            tensor([1.0]),
            1,
            _exact_scalar_decay,
            False,
            _scalar_decay_jacobian_rhs(),
        ),
        (
            "split_decay1",
            tensor([1.0]),
            1,
            _exact_scalar_decay,
            False,
            _ti.SplitRHS(split_explicit, split_implicit, split_jacobian),
        ),
        (
            "semilinear1",
            tensor([1.0]),
            1,
            _exact_semilinear,
            False,
            _ti.SemilinearRHS(tensor([[-2.0]]), semilinear_forcing),
        ),
        (
            "osc2",
            tensor([1.0, 0.0]),
            2,
            _exact_osc,
            False,
            _ti.CompositeRHS([_ti.BlackBoxRHS(fA), _ti.BlackBoxRHS(fB)]),
        ),
        (
            "ham2",
            tensor([1.0, 0.0]),
            2,
            _exact_ham,
            False,
            _ti.HamiltonianRHS(dT_dp=lambda p: p, dV_dq=lambda q: q, split_index=1),
        ),
    ]


_ORDERS = [1, 2, 3, 4, 5, 6]
_PROBS = _build_probs()


def _prob_on_backend(pid: str, backend: Any) -> tuple:
    for prob in _build_probs(backend):
        if prob[0] == pid:
            return prob
    raise KeyError(pid)


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
class _PerformanceSpec:
    name: str
    run: Callable[[ExecutionPlan], tuple[float, float, int, float]]
    tol: float


class _ConvergenceClaim(Claim[ExecutionPlan]):
    """Convergence + conservation claim for one (order, problem) pair."""

    def __init__(self, order: int, prob: tuple) -> None:
        self._order = order
        self._prob = prob

    @property
    def description(self) -> str:
        return f"convergence/order{self._order}/{self._prob[0]}"

    def check(self, execution_plan: ExecutionPlan) -> None:
        pid = self._prob[0]
        pid, u0, n, exact, mass_cons, rhs = _prob_on_backend(
            pid, execution_plan.backend
        )
        inst = _ti.AutoIntegrator(self._order)
        dts: list[float] = []
        errs: list[float] = []
        dt = 0.1
        state: _ti.ODEState | None = None
        refinements = execution_plan.refinement_count_for(
            work_fmas=lambda count: count * execution_plan.fma_rate / 8.0,
            min_refinements=4,
            max_refinements=8,
            label=self.description,
            safety=0.5,
        )
        for _ in range(refinements):
            try:
                state = _run(inst, rhs, u0, dt)
            except ValueError as e:
                pytest.skip(str(e))
            state.u.sync()
            errs.append(_err(state.u, exact, state.t))
            dts.append(dt)
            dt /= 2.0
            if errs[-1] == 0.0:
                break
        assert (
            _slope(errs, dts) >= self._order - 0.3
        ), f"order{self._order}/{pid}/{execution_plan.device_kind}: slope too low"
        if mass_cons and state is not None:
            label = f"order{self._order}/{pid}/{execution_plan.device_kind}"
            assert _conserved(state.u, n), f"{label}: mass not conserved"
            for i in range(n):
                assert float(state.u[i]) >= -1e-10, f"{label}: u[{i}] negative"


class _CorrectnessClaim(Claim[Any]):
    """Accuracy claim for one numerical history against an analytical f(t)."""

    def __init__(self, spec: _CorrectnessSpec) -> None:
        self._spec = spec

    @property
    def description(self) -> str:
        return f"correctness/{self._spec.name}"

    @property
    def expected_walltime_s(self) -> float:
        return self._spec.expected_walltime_s

    def check(self, _calibration: Any) -> None:
        self.skip_if_over_walltime_budget()
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


class _DomainClaim(Claim[Any]):
    """Correctness claim for state-domain predicates."""

    def __init__(self, name: str, check: Callable[[], None]) -> None:
        self._name = name
        self._check = check

    @property
    def description(self) -> str:
        return f"correctness/domain/{self._name}"

    def check(self, _calibration: Any) -> None:
        self._check()


def _domain_claims() -> list[_DomainClaim]:
    def _accepts_nonnegative_state() -> None:
        domain = _ti.NonnegativeStateDomain(3, roundoff_tolerance=1e-14)

        result = domain.check(Tensor([0.0, 0.25, 1.0], backend=_TIME_BACKEND))

        assert result.accepted
        assert result.violation is None

    def _accepts_roundoff_negative_state() -> None:
        domain = _ti.NonnegativeStateDomain(2, roundoff_tolerance=1e-12)

        result = domain.check(Tensor([1.0, -5e-13], backend=_TIME_BACKEND))

        assert result.accepted

    def _rejects_material_negative_state() -> None:
        domain = _ti.NonnegativeStateDomain(3, roundoff_tolerance=1e-12)

        result = domain.check(Tensor([0.1, -1e-9, -2e-9], backend=_TIME_BACKEND))

        assert result.rejected
        assert result.violation is not None
        assert result.violation.component == 2
        assert result.violation.value == -2e-9
        assert result.violation.tolerance == 1e-12
        assert result.violation.margin > 0.0

    def _rejects_wrong_shape() -> None:
        domain = _ti.NonnegativeStateDomain(3)

        result = domain.check(Tensor([[1.0, 0.0, 0.0]], backend=_TIME_BACKEND))

        assert result.rejected
        assert result.violation is not None
        assert result.violation.component is None
        assert "shape" in result.violation.reason

    def _reaction_network_exposes_abundance_domain() -> None:
        rhs = _ti.ReactionNetworkRHS(
            Tensor([[-1.0], [1.0]], backend=_TIME_BACKEND),
            lambda t, u: Tensor([float(u[0])], backend=u.backend),
            lambda t, u: Tensor([float(u[1])], backend=u.backend),
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
        )

        assert rhs.state_domain.check(
            Tensor([0.25, 0.75], backend=_TIME_BACKEND)
        ).accepted
        result = rhs.state_domain.check(Tensor([0.25, -1e-6], backend=_TIME_BACKEND))

        assert result.rejected
        assert result.violation is not None
        assert result.violation.component == 1

    def _auto_selects_capability_registry_branches() -> None:
        expected = {
            "base2": "RungeKuttaIntegrator",
            "jac_decay1": "ImplicitRungeKuttaIntegrator",
            "split_decay1": "AdditiveRungeKuttaIntegrator",
            "semilinear1": "LawsonRungeKuttaIntegrator",
            "osc2": "CompositionIntegrator",
            "ham2": "SymplecticCompositionIntegrator",
        }
        auto = _ti.AutoIntegrator(4)
        by_name = {name: rhs for name, _u0, _n, _exact, _mass, rhs in _PROBS}
        for name, implementation in expected.items():
            assert auto.select(by_name[name]).implementation == implementation

        with pytest.raises(ValueError, match="no algorithm"):
            _ti.AutoIntegrator(3).select(by_name["ham2"])

    def _predicts_time_to_nonnegative_boundary() -> None:
        domain = _ti.NonnegativeStateDomain(3, roundoff_tolerance=1e-12)

        limit = domain.step_limit(
            Tensor([1.0, 0.25, 0.0], backend=_TIME_BACKEND),
            Tensor([-2.0, -0.25, 1.0], backend=_TIME_BACKEND),
            safety=0.5,
        )

        assert limit is not None
        assert abs(limit - 0.25) < 1e-12

    return [
        _DomainClaim("nonnegative_accepts_valid", _accepts_nonnegative_state),
        _DomainClaim("nonnegative_accepts_roundoff", _accepts_roundoff_negative_state),
        _DomainClaim("nonnegative_rejects_negative", _rejects_material_negative_state),
        _DomainClaim("nonnegative_rejects_shape", _rejects_wrong_shape),
        _DomainClaim(
            "nonnegative_predicts_boundary_limit",
            _predicts_time_to_nonnegative_boundary,
        ),
        _DomainClaim(
            "reaction_network_exposes_abundance_domain",
            _reaction_network_exposes_abundance_domain,
        ),
        _DomainClaim(
            "auto_selects_capability_registry_branches",
            _auto_selects_capability_registry_branches,
        ),
    ]


class _BatchedDecayCorrectnessClaim(Claim[ExecutionPlan]):
    """Accuracy claim for many independent scalar decays in one Tensor state."""

    _DT = 0.02
    _T_END = 0.4
    _ORDER = 4
    _TOL = 1e-6
    _FMAS_PER_LANE_STEP = 64.0

    @property
    def description(self) -> str:
        return "correctness/batched_tensor/rk4_decay"

    def check(self, execution_plan: ExecutionPlan) -> None:
        steps = round(self._T_END / self._DT)
        max_batch = 512 if execution_plan.device_kind == "gpu" else 32
        source_batch = execution_plan.batch_size_for(
            fmas_per_case=steps * self._ORDER * self._FMAS_PER_LANE_STEP,
            min_batch=4,
            max_batch=max_batch,
        )
        lane_indices = execution_plan.batch_indices_for(
            source_batch, label=self.description
        )
        backend = execution_plan.backend
        rates_values = [0.2 + 1.8 * i / max(source_batch - 1, 1) for i in lane_indices]
        rates = Tensor(rates_values, backend=backend)
        u0_values = [1.0 + 0.1 * math.sin(i) for i in lane_indices]
        u0 = Tensor(u0_values, backend=backend)
        rhs = _ti.BlackBoxRHS(lambda t, u: -(rates * u))
        state = _run(
            _ti.RungeKuttaIntegrator(self._ORDER), rhs, u0, self._DT, self._T_END
        )
        state.u.sync()
        actual = state.u.to_list()
        expected = [
            u0_i * math.exp(-rate_i * state.t)
            for u0_i, rate_i in zip(u0_values, rates_values, strict=True)
        ]
        errors = [abs(float(a) - e) for a, e in zip(actual, expected, strict=True)]
        local_worst_i, worst_error = max(enumerate(errors), key=lambda item: item[1])
        batch_index = lane_indices[local_worst_i]
        assert worst_error < self._TOL, BatchedFailure(
            claim=self.description,
            device_kind=execution_plan.device_kind,
            batch_size=source_batch,
            batch_index=batch_index,
            method="RungeKuttaIntegrator",
            order=self._ORDER,
            problem="scalar_decay",
            parameters={
                "dt": self._DT,
                "t_end": self._T_END,
                "rate": rates_values[local_worst_i],
                "u0": u0_values[local_worst_i],
            },
            actual=actual[local_worst_i],
            expected=expected[local_worst_i],
            error=worst_error,
            tolerance=self._TOL,
        ).format()


class _BatchedOscillatorCorrectnessClaim(Claim[ExecutionPlan]):
    """Accuracy claim for independent harmonic oscillators in one Tensor state."""

    _DT = 0.01
    _T_END = 0.4
    _ORDER = 4
    _TOL = 1e-6
    _FMAS_PER_LANE_STEP = 128.0

    @property
    def description(self) -> str:
        return "correctness/batched_tensor/rk4_oscillator"

    def check(self, execution_plan: ExecutionPlan) -> None:
        steps = round(self._T_END / self._DT)
        max_batch = 512 if execution_plan.device_kind == "gpu" else 32
        source_batch = execution_plan.batch_size_for(
            fmas_per_case=steps * self._ORDER * self._FMAS_PER_LANE_STEP,
            min_batch=4,
            max_batch=max_batch,
        )
        lane_indices = execution_plan.batch_indices_for(
            source_batch, label=self.description
        )
        batch = len(lane_indices)
        backend = execution_plan.backend
        matrix = [[0.0 for _ in range(2 * batch)] for _ in range(2 * batch)]
        for i in range(batch):
            matrix[i][batch + i] = -1.0
            matrix[batch + i][i] = 1.0
        q0 = [1.0 + 0.05 * math.sin(i) for i in lane_indices]
        p0 = [0.1 * math.cos(i) for i in lane_indices]
        u0 = Tensor(q0 + p0, backend=backend)
        operator = Tensor(matrix, backend=backend)
        rhs = _ti.BlackBoxRHS(lambda t, u: operator @ u)
        state = _run(
            _ti.RungeKuttaIntegrator(self._ORDER), rhs, u0, self._DT, self._T_END
        )
        state.u.sync()
        actual = state.u.to_list()
        cos_t = math.cos(state.t)
        sin_t = math.sin(state.t)
        errors = []
        expected_pairs = []
        actual_pairs = []
        for i, (q_initial, p_initial) in enumerate(zip(q0, p0, strict=True)):
            expected_q = q_initial * cos_t - p_initial * sin_t
            expected_p = q_initial * sin_t + p_initial * cos_t
            actual_q = float(actual[i])
            actual_p = float(actual[batch + i])
            expected_pairs.append((expected_q, expected_p))
            actual_pairs.append((actual_q, actual_p))
            errors.append(max(abs(actual_q - expected_q), abs(actual_p - expected_p)))
        local_worst_i, worst_error = max(enumerate(errors), key=lambda item: item[1])
        batch_index = lane_indices[local_worst_i]
        assert worst_error < self._TOL, BatchedFailure(
            claim=self.description,
            device_kind=execution_plan.device_kind,
            batch_size=source_batch,
            batch_index=batch_index,
            method="RungeKuttaIntegrator",
            order=self._ORDER,
            problem="harmonic_oscillator",
            parameters={
                "dt": self._DT,
                "t_end": self._T_END,
                "q0": q0[local_worst_i],
                "p0": p0[local_worst_i],
            },
            actual=actual_pairs[local_worst_i],
            expected=expected_pairs[local_worst_i],
            error=worst_error,
            tolerance=self._TOL,
        ).format()


class _BatchedStiffDecayCorrectnessClaim(Claim[ExecutionPlan]):
    """Accuracy claim for independent IMEX stiff decays in one Tensor state."""

    _DT = 0.005
    _T_END = 0.1
    _ORDER = 2
    _TOL = 5e-4
    _FMAS_PER_LANE_STEP = 512.0

    @property
    def description(self) -> str:
        return "correctness/batched_tensor/imex_stiff_decay"

    def check(self, execution_plan: ExecutionPlan) -> None:
        steps = round(self._T_END / self._DT)
        max_batch = 128 if execution_plan.device_kind == "gpu" else 16
        source_batch = execution_plan.batch_size_for(
            fmas_per_case=steps * self._ORDER * self._FMAS_PER_LANE_STEP,
            min_batch=4,
            max_batch=max_batch,
        )
        lane_indices = execution_plan.batch_indices_for(
            source_batch, label=self.description
        )
        backend = execution_plan.backend
        rates = [8.0 + 24.0 * i / max(source_batch - 1, 1) for i in lane_indices]
        u0_values = [1.0 + 0.05 * math.sin(i) for i in lane_indices]
        explicit = Tensor(
            [
                [-0.2 * rate if i == j else 0.0 for j, rate in enumerate(rates)]
                for i in range(len(rates))
            ],
            backend=backend,
        )
        implicit = Tensor(
            [
                [-0.8 * rate if i == j else 0.0 for j, rate in enumerate(rates)]
                for i in range(len(rates))
            ],
            backend=backend,
        )
        rhs = _ti.SplitRHS(
            lambda t, u: explicit @ u,
            lambda t, u: implicit @ u,
            lambda t, u: implicit,
        )
        state = _run(
            _ti.AdditiveRungeKuttaIntegrator(self._ORDER),
            rhs,
            Tensor(u0_values, backend=backend),
            self._DT,
            self._T_END,
        )
        state.u.sync()
        actual = state.u.to_list()
        expected = [
            u0_i * math.exp(-rate_i * state.t)
            for u0_i, rate_i in zip(u0_values, rates, strict=True)
        ]
        errors = [abs(float(a) - e) for a, e in zip(actual, expected, strict=True)]
        local_worst_i, worst_error = max(enumerate(errors), key=lambda item: item[1])
        batch_index = lane_indices[local_worst_i]
        assert worst_error < self._TOL, BatchedFailure(
            claim=self.description,
            device_kind=execution_plan.device_kind,
            batch_size=source_batch,
            batch_index=batch_index,
            method="AdditiveRungeKuttaIntegrator",
            order=self._ORDER,
            problem="stiff_decay",
            parameters={
                "dt": self._DT,
                "t_end": self._T_END,
                "rate": rates[local_worst_i],
                "u0": u0_values[local_worst_i],
            },
            actual=actual[local_worst_i],
            expected=expected[local_worst_i],
            error=worst_error,
            tolerance=self._TOL,
        ).format()


class _BatchedSemilinearCorrectnessClaim(Claim[ExecutionPlan]):
    """Accuracy claim for independent Lawson semilinear forcing lanes."""

    _DT = 0.02
    _T_END = 0.4
    _ORDER = 4
    _TOL = 1e-6
    _FMAS_PER_LANE_STEP = 384.0

    @property
    def description(self) -> str:
        return "correctness/batched_tensor/lawson_semilinear_forcing"

    def check(self, execution_plan: ExecutionPlan) -> None:
        steps = round(self._T_END / self._DT)
        max_batch = 128 if execution_plan.device_kind == "gpu" else 16
        source_batch = execution_plan.batch_size_for(
            fmas_per_case=steps * self._ORDER * self._FMAS_PER_LANE_STEP,
            min_batch=4,
            max_batch=max_batch,
        )
        lane_indices = execution_plan.batch_indices_for(
            source_batch, label=self.description
        )
        backend = execution_plan.backend
        batch = len(lane_indices)
        linear = Tensor(
            [[-2.0 if i == j else 0.0 for j in range(batch)] for i in range(batch)],
            backend=backend,
        )
        u0_values = [1.0 + 0.05 * math.sin(i) for i in lane_indices]
        rhs = _ti.SemilinearRHS(
            linear,
            lambda t, u: Tensor([math.sin(t)] * batch, backend=u.backend),
        )
        state = _run(
            _ti.LawsonRungeKuttaIntegrator(self._ORDER),
            rhs,
            Tensor(u0_values, backend=backend),
            self._DT,
            self._T_END,
        )
        state.u.sync()
        actual = state.u.to_list()
        forced = _exact_semilinear(state.t)[0] - math.exp(-2.0 * state.t)
        expected = [u0_i * math.exp(-2.0 * state.t) + forced for u0_i in u0_values]
        errors = [abs(float(a) - e) for a, e in zip(actual, expected, strict=True)]
        local_worst_i, worst_error = max(enumerate(errors), key=lambda item: item[1])
        batch_index = lane_indices[local_worst_i]
        assert worst_error < self._TOL, BatchedFailure(
            claim=self.description,
            device_kind=execution_plan.device_kind,
            batch_size=source_batch,
            batch_index=batch_index,
            method="LawsonRungeKuttaIntegrator",
            order=self._ORDER,
            problem="semilinear_forcing",
            parameters={
                "dt": self._DT,
                "t_end": self._T_END,
                "u0": u0_values[local_worst_i],
            },
            actual=actual[local_worst_i],
            expected=expected[local_worst_i],
            error=worst_error,
            tolerance=self._TOL,
        ).format()


class _BatchedAdaptiveDecayCorrectnessClaim(Claim[ExecutionPlan]):
    """Accuracy claim for an embedded RK method driven by PIController."""

    _T_END = 0.4
    _ORDER = 5
    _TOL = 5e-6
    _FMAS_PER_LANE = 4096.0

    @property
    def description(self) -> str:
        return "correctness/batched_tensor/pi_adaptive_decay"

    def check(self, execution_plan: ExecutionPlan) -> None:
        max_batch = 128 if execution_plan.device_kind == "gpu" else 16
        source_batch = execution_plan.batch_size_for(
            fmas_per_case=self._FMAS_PER_LANE,
            min_batch=4,
            max_batch=max_batch,
        )
        lane_indices = execution_plan.batch_indices_for(
            source_batch, label=self.description
        )
        backend = execution_plan.backend
        rates = [0.2 + 1.8 * i / max(source_batch - 1, 1) for i in lane_indices]
        u0_values = [1.0 + 0.1 * math.sin(i) for i in lane_indices]
        matrix = Tensor(
            [
                [-rate if i == j else 0.0 for j, rate in enumerate(rates)]
                for i in range(len(rates))
            ],
            backend=backend,
        )
        rhs = _ti.BlackBoxRHS(lambda t, u: matrix @ u)
        controller = _ti.PIController(
            alpha=0.7 / self._ORDER,
            beta=0.4 / self._ORDER,
            tol=self._TOL,
            dt0=0.05,
            factor_max=2.0,
        )
        state = _ti.IntegrationDriver(
            _ti.RungeKuttaIntegrator(self._ORDER), controller=controller
        ).advance(rhs, Tensor(u0_values, backend=backend), 0.0, self._T_END)
        state.u.sync()
        actual = state.u.to_list()
        expected = [
            u0_i * math.exp(-rate_i * state.t)
            for u0_i, rate_i in zip(u0_values, rates, strict=True)
        ]
        errors = [abs(float(a) - e) for a, e in zip(actual, expected, strict=True)]
        local_worst_i, worst_error = max(enumerate(errors), key=lambda item: item[1])
        batch_index = lane_indices[local_worst_i]
        assert worst_error < self._TOL, BatchedFailure(
            claim=self.description,
            device_kind=execution_plan.device_kind,
            batch_size=source_batch,
            batch_index=batch_index,
            method="RungeKuttaIntegrator+PIController",
            order=self._ORDER,
            problem="adaptive_scalar_decay",
            parameters={
                "t_end": self._T_END,
                "rate": rates[local_worst_i],
                "u0": u0_values[local_worst_i],
            },
            actual=actual[local_worst_i],
            expected=expected[local_worst_i],
            error=worst_error,
            tolerance=self._TOL,
        ).format()


class _PerformanceClaim(Claim[ExecutionPlan]):
    """Cost-to-accuracy claim against the execution plan's Tensor roofline."""

    def __init__(self, spec: _PerformanceSpec) -> None:
        self._spec = spec

    @property
    def description(self) -> str:
        return f"performance/{self._spec.name}"

    def check(self, execution_plan: ExecutionPlan) -> None:
        elapsed, err, extent, fmas = self._spec.run(execution_plan)
        assert err < self._spec.tol, (
            f"{self.description}/{execution_plan.device_kind}: "
            f"extent={extent}, error={err:.3e} >= {self._spec.tol:.3e}"
        )
        if execution_plan.device_kind == "gpu" and fmas < _GPU_MIN_PERF_FMAS:
            pytest.skip(
                f"{self.description}/gpu: extent={extent} gives {fmas:.3g} FMAs, "
                f"below compute-bound threshold {_GPU_MIN_PERF_FMAS:.3g}"
            )
        roofline = fmas / execution_plan.fma_rate
        ratio = elapsed / roofline
        assert elapsed <= _PERF_OVERHEAD * roofline, (
            f"{self.description}/{execution_plan.device_kind}: "
            f"extent={extent}, fmas={fmas:.3g}, "
            f"{elapsed:.3e}s actual, {roofline:.3e}s Tensor roofline, "
            f"ratio={ratio:.1f}x > {_PERF_OVERHEAD:.1f}x"
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
        return Tensor([1.0] + [0.0] * self.p, backend=_TIME_BACKEND)

    def build_rhs(self) -> _ti.ReactionNetworkRHS:
        n, p, rates, topo = self.n, self.p, list(self.rates), self.topo
        rows = [[0.0] * p for _ in range(n)]
        for j in range(p):
            rows[j if topo == "chain" else 0][j] = -1.0
            rows[j + 1][j] = 1.0
        S = Tensor(rows, backend=_TIME_BACKEND)

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
    batched_names = {"jac_decay1", "split_decay1", "semilinear1", "osc2"}
    for name, u0, _n, exact, _mass, rhs in _PROBS:
        if name in batched_names:
            continue
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
        if spec.topo == "chain":
            ctrl = _adaptive_nordsieck_controller()
            return [ctrl.advance(rhs, spec.u0(), 0.0, spec.t_end(), spec.dt0())]
        ctrl = _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=_ti.ImplicitRungeKuttaIntegrator(2),
            inner=_ti.PIController(
                alpha=0.35,
                beta=0.2,
                tol=1e-5,
                dt0=spec.dt0(),
                factor_max=1.15,
            ),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        return [ctrl.advance(spec.u0(), 0.0, spec.t_end(), stop_at_nse=True)]

    def expected(t: float) -> tuple[float, ...]:
        return (1.0 / spec.n,) * spec.n

    return _CorrectnessSpec(
        f"nse/{spec.name}", run, expected, 1e-6, expected_walltime_s
    )


def _nse_transient_correctness_spec() -> _CorrectnessSpec:
    k = 5.0

    def run() -> list[_ti.ODEState]:
        rhs = _ti.ReactionNetworkRHS(
            Tensor([[-1.0], [1.0]], backend=_TIME_BACKEND),
            lambda t, u: Tensor([k * float(u[0])], backend=u.backend),
            lambda t, u: Tensor([k * float(u[1])], backend=u.backend),
            Tensor([0.9, 0.1], backend=_TIME_BACKEND),
        )
        ctrl = _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=_ti.ImplicitRungeKuttaIntegrator(2),
            inner=_ti.PIController(
                alpha=0.35,
                beta=0.2,
                tol=1e-5,
                dt0=0.01,
                factor_max=1.15,
            ),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        return [ctrl.advance(Tensor([0.7, 0.3], backend=_TIME_BACKEND), 0.0, 0.1)]

    def expected(t: float) -> tuple[float, float]:
        x0 = 0.5 + 0.2 * math.exp(-2.0 * k * t)
        return x0, 1.0 - x0

    return _CorrectnessSpec("nse/transient", run, expected, 1e-4)


RateFn = Callable[[float], list[tuple[int, int, float]]]


def _linear_network_rhs(rate_fn: RateFn, n: int) -> _ti.ReactionNetworkRHS:
    """Build a mass-conserving linear reaction network RHS from edge rates."""
    edges0 = rate_fn(0.0)
    stoich = [[0.0 for _ in edges0] for _ in range(n)]
    for j, (src, dst, _rate) in enumerate(edges0):
        stoich[src][j] = -1.0
        stoich[dst][j] = 1.0

    def forward_rate(t: float, u: Tensor) -> Tensor:
        return Tensor(
            [rate * float(u[src]) for src, _dst, rate in rate_fn(t)],
            backend=u.backend,
        )

    def reverse_rate(t: float, u: Tensor) -> Tensor:
        return Tensor([0.0 for _ in edges0], backend=u.backend)

    def jac(t: float, u: Tensor) -> Tensor:
        mat = [[0.0 for _ in range(n)] for _ in range(n)]
        for src, dst, rate in rate_fn(t):
            mat[src][src] -= rate
            mat[dst][src] += rate
        return Tensor(mat, backend=u.backend)

    return _ti.ReactionNetworkRHS(
        Tensor(stoich, backend=_TIME_BACKEND),
        forward_rate,
        reverse_rate,
        Tensor([1.0] + [0.0] * (n - 1), backend=_TIME_BACKEND),
        jac=jac,
    )


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


def _adaptive_nordsieck_controller(
    *, q_max: int = 6
) -> _ti.AdaptiveNordsieckController:
    return _ti.AdaptiveNordsieckController(
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
        rhs = _linear_network_rhs(_alpha_chain_rates(13), 13)
        controller = _adaptive_nordsieck_controller()
        state = controller.advance(
            rhs,
            Tensor([1.0] + [0.0] * 12, backend=_TIME_BACKEND),
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
        "stress/adaptive_nordsieck_alpha_chain_rate_contrast",
        run,
        lambda t: (1.0,) * 13,
        float("inf"),
        expected_walltime_s=20.0,
    )


def _branched_hot_window_stress_spec() -> _CorrectnessSpec:
    def run() -> list[_ti.ODEState]:
        rhs = _linear_network_rhs(_branched_hot_window_rates(16), 16)
        coarse = _adaptive_nordsieck_controller()
        fine = _adaptive_nordsieck_controller()
        u0 = Tensor([1.0] + [0.0] * 15, backend=_TIME_BACKEND)
        coarse_state = coarse.advance(rhs, u0, t0=0.0, t_end=0.32, dt0=5e-4)
        fine_state = fine.advance(rhs, u0, t0=0.0, t_end=0.32, dt0=2.5e-4)
        _assert_abundance_state(coarse_state.u, label="branched_hot_window/coarse")
        _assert_abundance_state(fine_state.u, label="branched_hot_window/fine")
        assert "adams" in coarse.accepted_families
        assert "bdf" in coarse.accepted_families
        assert coarse.family_switches >= 1
        assert max(coarse.accepted_stiffness) > 1.0
        assert coarse.rejection_reasons.count("domain") > 0
        assert coarse.rejected_steps < 80
        assert float(norm(coarse_state.u - fine_state.u)) < 5e-2
        return [coarse_state]

    return _CorrectnessSpec(
        "stress/adaptive_nordsieck_branched_hot_window_self_consistency",
        run,
        lambda t: (1.0,) * 16,
        float("inf"),
        expected_walltime_s=30.0,
    )


def _adaptive_nordsieck_domain_rejection_spec() -> _CorrectnessSpec:
    def run() -> list[_ti.ODEState]:
        rate = 300.0
        rhs = _two_species_decay_rhs(rate)
        controller = _adaptive_nordsieck_controller()
        state = controller.advance(
            rhs,
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
            t0=0.0,
            t_end=0.03,
            dt0=0.005,
        )

        _assert_abundance_state(state.u, label="adaptive_nordsieck_domain_retry")
        assert controller.domain_limited_step_sizes
        assert max(controller.domain_limited_step_sizes) < 0.005
        assert controller.rejection_reasons.count("domain") == 0
        assert controller.rejected_steps < 20
        return [state]

    return _CorrectnessSpec(
        "domain/adaptive_nordsieck_limits_negative_abundance",
        run,
        lambda t: (1.0, 1.0),
        float("inf"),
    )


def _two_species_decay_rhs(rate: float) -> _ti.ReactionNetworkRHS:
    return _ti.ReactionNetworkRHS(
        Tensor([[-1.0], [1.0]], backend=_TIME_BACKEND),
        lambda t, u: Tensor([rate * float(u[0])], backend=u.backend),
        lambda t, u: Tensor([0.0], backend=u.backend),
        Tensor([1.0, 0.0], backend=_TIME_BACKEND),
        jac=lambda t, u: Tensor([[-rate, 0.0], [rate, 0.0]], backend=u.backend),
    )


def _generic_integrator_domain_rejection_spec() -> _CorrectnessSpec:
    def run() -> list[_ti.ODEState]:
        rate = 300.0
        rhs = _two_species_decay_rhs(rate)
        stepper = _ti.IntegrationDriver(
            _ti.RungeKuttaIntegrator(3),
            controller=_ti.PIController(
                alpha=0.35,
                beta=0.2,
                tol=1e-3,
                dt0=0.005,
            ),
        )
        state = stepper.advance(
            rhs,
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
            0.0,
            0.03,
        )

        _assert_abundance_state(state.u, label="generic_integrator_domain_retry")
        assert stepper.domain_limited_step_sizes
        assert max(stepper.domain_limited_step_sizes) < 0.005
        assert stepper.rejection_reasons.count("domain") == 0
        assert stepper.rejected_steps < 20
        return [state]

    return _CorrectnessSpec(
        "domain/generic_integrator_limits_negative_abundance",
        run,
        lambda t: (1.0, 1.0),
        float("inf"),
    )


def _constraint_aware_domain_rejection_spec() -> _CorrectnessSpec:
    def run() -> list[_ti.ODEState]:
        rhs = _two_species_decay_rhs(1000.0)
        controller = _ti.ConstraintAwareController(
            rhs,
            integrator=_ti.ImplicitRungeKuttaIntegrator(2),
            inner=_ti.PIController(
                alpha=0.35,
                beta=0.2,
                tol=1.0,
                dt0=0.02,
            ),
        )
        state = controller.advance(
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
            0.0,
            0.03,
        )

        _assert_abundance_state(state.u, label="constraint_aware_domain_retry")
        assert controller.domain_limited_step_sizes
        assert max(controller.domain_limited_step_sizes) < 0.02
        assert controller.rejection_reasons.count("domain") == 0
        assert controller.rejected_steps < 10
        return [state]

    return _CorrectnessSpec(
        "domain/constraint_aware_limits_negative_abundance",
        run,
        lambda t: (1.0, 1.0),
        float("inf"),
    )


def _explicit_rk4_performance_spec() -> _PerformanceSpec:
    dt, t_end, order = 0.02, 1.0, 4
    n_steps = round(t_end / dt)

    def run(execution_plan: ExecutionPlan) -> tuple[float, float, int, float]:
        max_batch = 512 if execution_plan.device_kind == "gpu" else 32
        batch = execution_plan.batch_size_for(
            fmas_per_case=n_steps * order * 64.0,
            min_batch=4,
            max_batch=max_batch,
        )
        backend = execution_plan.backend
        rates = Tensor(
            [0.2 + 1.8 * i / max(batch - 1, 1) for i in range(batch)],
            backend=backend,
        )
        u0_values = [1.0 + 0.1 * math.sin(i) for i in range(batch)]
        rhs = _ti.BlackBoxRHS(lambda t, u: -(rates * u))

        def execute() -> _ti.ODEState:
            state = _run(
                _ti.RungeKuttaIntegrator(order),
                rhs,
                Tensor(u0_values, backend=backend),
                dt,
                t_end,
            )
            state.u.sync()
            return state

        elapsed, state = _best_elapsed(execute)
        actual = state.u.to_list()
        expected = [
            u0_i * math.exp(-rate_i * state.t)
            for u0_i, rate_i in zip(u0_values, rates.to_list(), strict=True)
        ]
        err = max(abs(float(a) - e) for a, e in zip(actual, expected, strict=True))
        return elapsed, err, batch, batch * n_steps * order * 64.0

    return _PerformanceSpec(
        "explicit_rk4/scalar_decay",
        run,
        1e-7,
    )


def _semilinear_lawson4_performance_spec() -> _PerformanceSpec:
    dt, t_end, order = 0.02, 0.4, 4
    n_steps = round(t_end / dt)

    def run(execution_plan: ExecutionPlan) -> tuple[float, float, int, float]:
        max_batch = 128 if execution_plan.device_kind == "gpu" else 16
        batch = execution_plan.batch_size_for(
            fmas_per_case=n_steps * order * 384.0,
            min_batch=4,
            max_batch=max_batch,
        )
        backend = execution_plan.backend
        linear = Tensor(
            [[-2.0 if i == j else 0.0 for j in range(batch)] for i in range(batch)],
            backend=backend,
        )
        u0_values = [1.0 + 0.05 * math.sin(i) for i in range(batch)]
        rhs = _ti.SemilinearRHS(
            linear,
            lambda t, u: Tensor([math.sin(t)] * batch, backend=u.backend),
        )

        def execute() -> _ti.ODEState:
            state = _run(
                _ti.LawsonRungeKuttaIntegrator(order),
                rhs,
                Tensor(u0_values, backend=backend),
                dt,
                t_end,
            )
            state.u.sync()
            return state

        elapsed, state = _best_elapsed(execute)
        actual = state.u.to_list()
        forced = _exact_semilinear(state.t)[0] - math.exp(-2.0 * state.t)
        expected = [u0_i * math.exp(-2.0 * state.t) + forced for u0_i in u0_values]
        err = max(abs(float(a) - e) for a, e in zip(actual, expected, strict=True))
        return elapsed, err, batch, batch * n_steps * order * 384.0

    return _PerformanceSpec(
        "lawson_rk4/semilinear_forcing",
        run,
        1e-7,
    )


# ── claim registries ─────────────────────────────────────────────────────────

_CONV_CLAIMS: list[Claim[ExecutionPlan]] = [
    _ConvergenceClaim(order, prob) for order in _ORDERS for prob in _PROBS
]

_OFF_SPECS = _chain_specs(range(5, 12)) + _spoke_specs(range(7, 22), [1, 10, 100])
_CORRECT_CLAIMS: list[Claim[Any]] = [
    *_domain_claims(),
    _BatchedDecayCorrectnessClaim(),
    _BatchedOscillatorCorrectnessClaim(),
    _BatchedStiffDecayCorrectnessClaim(),
    _BatchedSemilinearCorrectnessClaim(),
    _BatchedAdaptiveDecayCorrectnessClaim(),
    *[_CorrectnessClaim(s) for s in _ode_correctness_specs()],
    *[_CorrectnessClaim(_nse_correctness_spec(s)) for s in _CI_SPECS],
    _CorrectnessClaim(_nse_transient_correctness_spec()),
    _CorrectnessClaim(_adaptive_nordsieck_domain_rejection_spec()),
    _CorrectnessClaim(_generic_integrator_domain_rejection_spec()),
    _CorrectnessClaim(_constraint_aware_domain_rejection_spec()),
    *[
        _CorrectnessClaim(_nse_correctness_spec(s, expected_walltime_s=5.0))
        for s in _OFF_SPECS
    ],
    _CorrectnessClaim(_alpha_chain_stress_spec()),
    _CorrectnessClaim(_branched_hot_window_stress_spec()),
]
_PERF_CLAIMS: list[Claim[ExecutionPlan]] = [
    _PerformanceClaim(_explicit_rk4_performance_spec()),
    _PerformanceClaim(_semilinear_lawson4_performance_spec()),
]


# ── parametric test functions (each body is a single claim.check() dispatch) ──


_CONV_IDS = [c.description for c in _CONV_CLAIMS]


@pytest.mark.parametrize("claim", _CONV_CLAIMS, ids=_CONV_IDS)
def test_convergence(
    claim: Claim[ExecutionPlan], execution_plan: ExecutionPlan
) -> None:
    claim.check(execution_plan)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[Any], execution_plan: ExecutionPlan) -> None:
    claim.check(execution_plan)


@pytest.mark.parametrize(
    "claim", _PERF_CLAIMS, ids=[c.description for c in _PERF_CLAIMS]
)
def test_performance(
    claim: Claim[ExecutionPlan], execution_plan: ExecutionPlan
) -> None:
    claim.check(execution_plan)
