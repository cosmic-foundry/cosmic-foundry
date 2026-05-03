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
from cosmic_foundry.computation.tensor import Tensor
from tests import time_integrator_cases as cases
from tests.claims import (
    BatchedFailure,
    Claim,
    ExecutionPlan,
)

_TIME_BACKEND = cases.TIME_BACKEND
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

    return [
        ("base2", tensor([1.0, 0.0]), 2, cases.exact2, True, _ti.BlackBoxRHS(f2)),
        (
            "decay3",
            tensor([1.0, 0.0, 0.0]),
            3,
            cases.exact3,
            True,
            _ti.BlackBoxRHS(f3),
        ),
        (
            "jac_decay1",
            tensor([1.0]),
            1,
            cases.exact_scalar_decay,
            False,
            cases.scalar_decay_jacobian_rhs(),
        ),
        (
            "split_decay1",
            tensor([1.0]),
            1,
            cases.exact_scalar_decay,
            False,
            cases.split_decay_rhs(),
        ),
        (
            "semilinear1",
            tensor([1.0]),
            1,
            cases.exact_semilinear,
            False,
            cases.semilinear_forcing_rhs(backend),
        ),
        (
            "osc2",
            tensor([1.0, 0.0]),
            2,
            cases.exact_osc,
            False,
            cases.oscillator_composite_rhs(),
        ),
        (
            "ham2",
            tensor([1.0, 0.0]),
            2,
            cases.exact_ham,
            False,
            cases.harmonic_hamiltonian_rhs(),
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
            errs.append(cases.err(state.u, exact, state.t))
            dts.append(dt)
            dt /= 2.0
            if errs[-1] == 0.0:
                break
        assert (
            _slope(errs, dts) >= self._order - 0.3
        ), f"order{self._order}/{pid}/{execution_plan.device_kind}: slope too low"
        if mass_cons and state is not None:
            label = f"order{self._order}/{pid}/{execution_plan.device_kind}"
            assert cases.conserved(state.u, n), f"{label}: mass not conserved"
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
            assert cases.err(state.u, self._spec.expected, state.t) < self._spec.tol


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
        forced = cases.exact_semilinear(state.t)[0] - math.exp(-2.0 * state.t)
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
        forced = cases.exact_semilinear(state.t)[0] - math.exp(-2.0 * state.t)
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

_CORRECT_CLAIMS: list[Claim[Any]] = [
    _BatchedDecayCorrectnessClaim(),
    _BatchedOscillatorCorrectnessClaim(),
    _BatchedStiffDecayCorrectnessClaim(),
    _BatchedSemilinearCorrectnessClaim(),
    _BatchedAdaptiveDecayCorrectnessClaim(),
    *[_CorrectnessClaim(s) for s in _ode_correctness_specs()],
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
