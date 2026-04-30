"""Verification for the time-integration layer — single parametric test function."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import pytest
import sympy

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.backends import (
    NumpyBackend,
    get_default_backend,
    set_default_backend,
)
from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators._newton import newton_solve
from cosmic_foundry.computation.time_integrators.integrator import TimeIntegrator

_PREV = get_default_backend()
set_default_backend(NumpyBackend())

_DT = 0.1
_NH = 5
_BDF_NH = 7
_PI_A, _PI_B = 0.7, 0.4
_CHAIN_N = range(3, 5)
_SPOKE_N = range(3, 7)
_SPOKE_K: list[int] = [1, 10]
_OFFLINE = os.environ.get("COSMIC_FOUNDRY_OFFLINE_NETWORK_STRESS") == "1"
_CHAIN_N_OFF = range(5, 12)
_SPOKE_N_OFF = range(7, 22)
_SPOKE_K_OFF: list[int] = [1, 10, 100]


@pytest.fixture(scope="module", autouse=True)
def _numpy_backend() -> Any:
    yield
    set_default_backend(_PREV)


@dataclass
class _NetworkSpec:
    name: str
    topology: str
    n_species: int
    rates: list[float]

    @property
    def n_pairs(self) -> int:
        return self.n_species - 1

    def t_end(self) -> float:
        k = min(self.rates)
        return 4.0 * self.n_pairs**2 / k if self.topology == "chain" else 8.0 / k

    def dt0(self) -> float:
        return min(0.05, 0.1 / max(self.rates))

    def abundance_tol(self) -> float:
        return 1e-7

    def u0(self) -> Tensor:
        return Tensor([1.0] + [0.0] * self.n_pairs)

    def build_rhs(self) -> _ti.ReactionNetworkRHS:
        n, p, rates, topo = (
            self.n_species,
            self.n_pairs,
            list(self.rates),
            self.topology,
        )
        rows = [[0.0] * p for _ in range(n)]
        if topo == "chain":
            for j in range(p):
                rows[j][j] = -1.0
                rows[j + 1][j] = 1.0
        else:
            for j in range(p):
                rows[0][j] = -1.0
                rows[j + 1][j] = 1.0
        S = Tensor(rows)
        if topo == "chain":

            def r_plus(t: float, u: Tensor) -> Tensor:
                return Tensor(
                    [rates[j] * float(u[j]) for j in range(p)], backend=u.backend
                )

            def r_minus(t: float, u: Tensor) -> Tensor:
                return Tensor(
                    [rates[j] * float(u[j + 1]) for j in range(p)], backend=u.backend
                )

        else:

            def r_plus(t: float, u: Tensor) -> Tensor:
                a0 = float(u[0])
                return Tensor([rates[j] * a0 for j in range(p)], backend=u.backend)

            def r_minus(t: float, u: Tensor) -> Tensor:
                return Tensor(
                    [rates[j] * float(u[j + 1]) for j in range(p)], backend=u.backend
                )

        return _ti.ReactionNetworkRHS(S, r_plus, r_minus, self.u0())


def _chain_specs(n_range: range, k: float = 1.0) -> list[_NetworkSpec]:
    return [
        _NetworkSpec(f"chain-n{n}-k{k:.0f}", "chain", n, [k] * (n - 1)) for n in n_range
    ]


def _spoke_specs(n_range: range, k_ratios: list[int]) -> list[_NetworkSpec]:
    specs = []
    for n in n_range:
        p, nf = n - 1, (n - 1) // 2
        for k in k_ratios:
            specs.append(
                _NetworkSpec(
                    f"spoke-n{n}-k{k:.0f}",
                    "spoke",
                    n,
                    [float(k)] * nf + [1.0] * (p - nf),
                )
            )
    return specs


def _nse_claim(spec: _NetworkSpec) -> pytest.param:  # type: ignore[type-arg]
    def _chk() -> None:
        rhs = spec.build_rhs()
        ctrl = _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=_ti.implicit_midpoint,
            inner=_ti.PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=spec.dt0()),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        state = ctrl.advance(spec.u0(), 0.0, spec.t_end())
        assert ctrl.nse_events, f"no NSE events for {spec.name}"
        assert state.active_constraints == frozenset(range(spec.n_pairs))
        eq = 1.0 / spec.n_species
        for i in range(spec.n_species):
            xi = float(state.u[i])
            assert (
                abs(xi - eq) < spec.abundance_tol()
            ), f"{spec.name}[{i}]: {abs(xi - eq):.3e}"

    return pytest.param(_chk, id=f"parametric_network/{spec.name}")


def calibrate_cost(n_s: int = 4, n_l: int = 8, reps: int = 50) -> tuple[float, float]:
    def _step(n: int) -> float:
        spec = _NetworkSpec(f"_cal_{n}", "chain", n, [1.0] * (n - 1))
        state = _ti.ODEState(0.0, spec.u0())
        t0 = time.perf_counter()
        for _ in range(reps):
            state = _ti.implicit_midpoint.step(spec.build_rhs(), state, 0.001)
        return (time.perf_counter() - t0) / reps

    c_s, c_l = _step(n_s), _step(n_l)
    denom = n_l**3 - n_s**3
    a = max((c_l - c_s) / denom, 0.0)
    b = max(c_s - a * n_s**3, 0.0)
    return a, b


def _cost_claim(spec: _NetworkSpec, cal: tuple[float, float]) -> pytest.param:  # type: ignore[type-arg]
    def _chk() -> None:
        ctrl = _ti.ConstraintAwareController(
            rhs=spec.build_rhs(),
            integrator=_ti.implicit_midpoint,
            inner=_ti.ConstantStep(dt=spec.dt0()),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        t0 = time.perf_counter()
        ctrl.advance(spec.u0(), 0.0, spec.t_end())
        actual = time.perf_counter() - t0
        pred = math.floor(spec.t_end() / spec.dt0()) * (
            cal[0] * spec.n_species**3 + cal[1]
        )
        assert actual < 2.0 * pred, f"{spec.name}: {actual:.3f}s > 2×{pred:.3f}s"

    return pytest.param(_chk, id=f"cost_model/{spec.name}")


def _build_params() -> list:  # noqa: PLR0912,PLR0914,PLR0915
    # ── shared constants ──────────────────────────────────────────────────────
    _k, _k1, _k2 = 1.0, 1.0, 2.0
    _U2, _U3 = Tensor([1.0, 0.0]), Tensor([1.0, 0.0, 0.0])
    _DK1 = Tensor([[-_k1, 0.0, 0.0], [_k1, -_k2, 0.0], [0.0, _k2, 0.0]])

    BASE_RHS = _ti.JacobianRHS(
        f=lambda t, u: Tensor([-_k * float(u[0]), _k * float(u[0])], backend=u.backend),
        jac=lambda t, u: Tensor([[-_k, 0.0], [_k, 0.0]], backend=u.backend),
    )
    BASE_IMEX = _ti.SplitRHS(
        f_E=lambda t, u: Tensor([0.0, _k * float(u[0])], backend=u.backend),
        f_I=lambda t, u: Tensor([-_k * float(u[0]), 0.0], backend=u.backend),
        jac_I=lambda t, u: Tensor([[-_k, 0.0], [0.0, 0.0]], backend=u.backend),
    )
    DECAY_RHS = _ti.JacobianRHS(
        f=lambda t, u: Tensor(
            [
                -_k1 * float(u[0]),
                _k1 * float(u[0]) - _k2 * float(u[1]),
                _k2 * float(u[1]),
            ],
            backend=u.backend,
        ),
        jac=lambda t, u: _DK1,
    )
    DECAY_IMEX = _ti.SplitRHS(
        f_E=lambda t, u: Tensor(
            [0.0, _k1 * float(u[0]), _k2 * float(u[1])], backend=u.backend
        ),
        f_I=lambda t, u: Tensor(
            [-_k1 * float(u[0]), -_k2 * float(u[1]), 0.0], backend=u.backend
        ),
        jac_I=lambda t, u: Tensor(
            [[-_k1, 0.0, 0.0], [0.0, -_k2, 0.0], [0.0, 0.0, 0.0]], backend=u.backend
        ),
    )

    def _stiff_f(t: float, u: Tensor) -> Tensor:
        k2 = 1.0 if t < 0.5 else 80.0
        return Tensor(
            [-float(u[0]), float(u[0]) - k2 * float(u[1]), k2 * float(u[1])],
            backend=u.backend,
        )

    def _stiff_jac(t: float, u: Tensor) -> Tensor:
        k2 = 1.0 if t < 0.5 else 80.0
        return Tensor([[-1, 0, 0], [1, -k2, 0], [0, k2, 0]])

    STIFFENING_RHS = _ti.JacobianRHS(f=_stiff_f, jac=_stiff_jac)

    def _vode_f(t: float, u: Tensor) -> Tensor:
        k2 = 1.0 if t < 0.45 else 1000.0
        return Tensor(
            [-float(u[0]), float(u[0]) - k2 * float(u[1]), k2 * float(u[1])],
            backend=u.backend,
        )

    def _vode_jac(t: float, u: Tensor) -> Tensor:
        k2 = 1.0 if t < 0.45 else 1000.0
        return Tensor([[-1, 0, 0], [1, -k2, 0], [0, k2, 0]])

    VODE_RHS = _ti.JacobianRHS(f=_vode_f, jac=_vode_jac)

    def _sharpening_f(t: float, u: Tensor) -> Tensor:
        k2 = 1.0 if t < 0.5 else 10.0
        return Tensor(
            [-float(u[0]), float(u[0]) - k2 * float(u[1]), k2 * float(u[1])],
            backend=u.backend,
        )

    def _base_err(u: Tensor, t: float) -> float:
        x0 = math.exp(-_k * t)
        return max(abs(float(u[0]) - x0), abs(float(u[1]) - (1.0 - x0)))

    def _decay_exact(t: float) -> tuple[float, float, float]:
        x0 = math.exp(-_k1 * t)
        x1 = math.exp(-_k1 * t) - math.exp(-_k2 * t)
        return x0, x1, 1.0 - x0 - x1

    def _decay_err(u: Tensor, t: float) -> float:
        x0, x1, x2 = _decay_exact(t)
        return max(abs(float(u[i]) - v) for i, v in enumerate((x0, x1, x2)))

    def _base_f(t: float, u: Tensor) -> Tensor:
        return Tensor([-_k * float(u[0]), _k * float(u[0])], backend=u.backend)

    def _decay_f(t: float, u: Tensor) -> Tensor:
        return Tensor(
            [
                -_k1 * float(u[0]),
                _k1 * float(u[0]) - _k2 * float(u[1]),
                _k2 * float(u[1]),
            ],
            backend=u.backend,
        )

    def _etd_rhs() -> _ti.SemilinearRHS:
        linear = Tensor([[-8.0, 0.0, 0.0], [8.0, -0.5, 0.0], [0.0, 0.5, 0.0]])

        def _res(t: float, u: Tensor) -> Tensor:
            rate = 0.25 + 0.1 * math.sin(2.0 * t)
            return Tensor(
                [-rate * float(u[0]), 0.0, rate * float(u[0])], backend=u.backend
            )

        return _ti.SemilinearRHS(linear, _res)

    ETD_RHS = _etd_rhs()

    def _split_a(t: float, u: Tensor) -> Tensor:
        return Tensor([-float(u[1]), 0.0], backend=u.backend)

    def _split_b(t: float, u: Tensor) -> Tensor:
        return Tensor([0.0, float(u[0])], backend=u.backend)

    SPLIT_RHS = _ti.CompositeRHS([_ti.BlackBoxRHS(_split_a), _ti.BlackBoxRHS(_split_b)])

    def _split_exact(t: float) -> Tensor:
        return Tensor([math.cos(t), math.sin(t)])

    def _integrate_etd(inst: object, dt: float, t_end: float = 0.5) -> _ti.ODEState:
        state = _ti.ODEState(0.0, _U3)
        for _ in range(round(t_end / dt)):
            state = inst.step(ETD_RHS, state, dt)  # type: ignore[union-attr]
        return state

    def _etd_cvg_fn(m: object) -> object:
        ref = _integrate_etd(m, 0.0015625)
        return lambda dt: float(norm(_integrate_etd(m, dt).u - ref.u))

    def _integrate_split(inst: object, dt: float, t_end: float = 1.0) -> _ti.ODEState:
        state = _ti.ODEState(0.0, _U2)
        for _ in range(round(t_end / dt)):
            state = inst.step(SPLIT_RHS, state, dt)  # type: ignore[union-attr]
        return state

    def _run_rk(m: object, rhs: object, u0: Tensor, dt: float) -> _ti.ODEState:
        state = _ti.ODEState(0.0, u0)
        for _ in range(math.ceil(1.0 / dt)):
            state = m.step(rhs, state, dt)  # type: ignore[union-attr]
        return state

    def _run_ms(m: object, rhs: object, u0: Tensor, dt: float) -> _ti.ODEState:
        order: int = m.order  # type: ignore[union-attr]
        state = m.init_state(rhs, 0.0, u0, dt)  # type: ignore[union-attr]
        for _ in range(max(math.ceil(1.0 / dt) - order, 1)):
            state = m.step(rhs, state, dt)  # type: ignore[union-attr]
        return state

    def _run_cons(
        m: object, rhs: object, u0: Tensor, dt: float, t_end: float
    ) -> _ti.ODEState:
        state = _ti.ODEState(0.0, u0)
        for _ in range(round(t_end / dt)):
            state = m.step(rhs, state, dt)  # type: ignore[union-attr]
        return state

    def _run_ms_cons(
        m: object, rhs: object, u0: Tensor, dt: float, t_end: float
    ) -> _ti.ODEState:
        order: int = m.order  # type: ignore[union-attr]
        state = m.init_state(rhs, 0.0, u0, dt)  # type: ignore[union-attr]
        for _ in range(max(round(t_end / dt) - order, 1)):
            state = m.step(rhs, state, dt)  # type: ignore[union-attr]
        return state

    def _run_sym(inst: object, dt: float) -> float:
        H = _ti.HamiltonianRHS(dT_dp=lambda p: p, dV_dq=lambda q: q, split_index=1)
        state = _ti.ODEState(0.0, _U2)
        for _ in range(round(1.0 / dt)):
            state = inst.step(H, state, dt)  # type: ignore[union-attr]
        return math.sqrt(
            (float(state.u[0]) - math.cos(state.t)) ** 2
            + (float(state.u[1]) - (-math.sin(state.t))) ** 2
        )

    def _slope(errors: list[float], dts: list[float], label: str = "") -> float:
        eps = sys.float_info.epsilon * 10
        valid = [(dt, e) for dt, e in zip(dts, errors, strict=False) if e > eps]
        if len(valid) < 3:
            raise AssertionError(
                f"{label + ': ' if label else ''}reached machine precision too early "
                f"({len(valid)} valid points above {eps:.1e})"
            )
        log_dts = [math.log(dt) for dt, _ in valid]
        log_errs = [math.log(e) for _, e in valid]
        n = len(log_dts)
        mx, my = sum(log_dts) / n, sum(log_errs) / n
        return sum(
            (x - mx) * (y - my) for x, y in zip(log_dts, log_errs, strict=False)
        ) / sum((x - mx) ** 2 for x in log_dts)

    def _assert_cons(u: Tensor, n: int, *, label: str, tol: float = 1e-12) -> None:
        total = sum(float(u[i]) for i in range(n))
        assert abs(total - 1.0) < tol, f"{label}: sum(X)={total:.15f}≠1"
        for i in range(n):
            assert float(u[i]) >= -1e-10, f"{label}: X[{i}]={float(u[i]):.3e}<0"

    # ── factory functions returning pytest.param ──────────────────────────────

    def _cvg(
        label: str,
        run_fn: object,
        order: int,
        dt_base: float = _DT,
        n_halvings: int = _NH,
        tol: float = 0.1,
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            dts = [dt_base / 2**k for k in range(n_halvings + 1)]
            errs = [run_fn(dt) for dt in dts]  # type: ignore[operator]
            s = _slope(errs, dts, label)
            assert s >= order - tol, f"{label}: slope {s:.3f} < {order}-{tol}"

        return pytest.param(_chk, id=f"convergence/{label}")

    def _cons(
        label: str,
        run_fn: object,
        n: int,
        exact_fn: object,
        dt: float = 0.05,
        t_end: float = 2.0,
        tol: float = 1e-3,
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            state = run_fn(dt, t_end)  # type: ignore[operator]
            exact = exact_fn(state.t)  # type: ignore[operator]
            err = max(abs(float(state.u[i]) - exact[i]) for i in range(n))
            assert err < tol, f"{label}: err {err:.2e} > {tol:.2e}"
            _assert_cons(state.u, n, label=label)

        return pytest.param(_chk, id=f"conservation/{label}")

    def _rk_ord(label: str, inst: object) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            for t in _ti.trees_up_to_order(inst.order):  # type: ignore[union-attr]
                alpha = _ti.elementary_weight(t, inst.A_sym, inst.b_sym)  # type: ignore[union-attr]
                exp = sympy.Rational(1) / _ti.gamma(t)
                assert sympy.simplify(alpha - exp) == 0, f"{label}: B-series failed {t}"

        return pytest.param(_chk, id=f"rk_order/{label}")

    def _stab_a(label: str, inst: object) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            R = _ti.stability_function(inst.A_sym, inst.b_sym)  # type: ignore[union-attr]
            z = sympy.Symbol("z")
            Rf = sympy.lambdify(z, R, modules="cmath")
            for w in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
                assert abs(Rf(complex(0, w))) <= 1.0 + 1e-10, f"{label}: |R(i{w})|>1"

        return pytest.param(_chk, id=f"a_stability/{label}")

    def _stab_l(label: str, inst: object) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            R = _ti.stability_function(inst.A_sym, inst.b_sym)  # type: ignore[union-attr]
            R_inf = abs(sympy.lambdify(sympy.Symbol("z"), R, modules="cmath")(-1e6))
            assert R_inf < 1e-4, f"{label}: |R(-1e6)|={R_inf:.2e}"

        return pytest.param(_chk, id=f"l_stability/{label}")

    def _stepper(
        inst: object, label: str, dt: float, t_end: float = 1.0, rtol: float = 1e-4
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            final = _ti.TimeStepper(inst, controller=_ti.ConstantStep(dt)).advance(
                _ti.BlackBoxRHS(_base_f), _U2, 0.0, t_end
            )
            assert abs(final.t - t_end) < 1e-12, f"{label}: t={final.t}"
            assert _base_err(final.u, t_end) < rtol, f"{label}: err>{rtol}"
            _assert_cons(final.u, 2, label=label)

        return pytest.param(_chk, id=f"stepper/{label}")

    def _pi_acc(
        inst: object,
        label: str,
        tol: float = 1e-4,
        t_end: float = 1.0,
        c_rel: float = 10.0,
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            p = inst.order  # type: ignore[union-attr]
            pi = _ti.PIController(alpha=_PI_A / p, beta=_PI_B / p, tol=tol, dt0=0.1)
            final = _ti.TimeStepper(inst, controller=pi).advance(
                _ti.BlackBoxRHS(_base_f), _U2, 0.0, t_end
            )
            err = _base_err(final.u, t_end)
            assert err <= c_rel * tol, f"{label}: {err:.2e}>{c_rel}×{tol:.2e}"
            _assert_cons(final.u, 2, label=label)

        return pytest.param(_chk, id=f"pi_accuracy/{label}")

    def _pi_wp(
        inst: object, label: str, tc: float = 1e-3, tf: float = 1e-5, t_end: float = 1.0
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            p = inst.order  # type: ignore[union-attr]

            def _run(tol: float) -> float:
                pi = _ti.PIController(alpha=_PI_A / p, beta=_PI_B / p, tol=tol, dt0=0.1)
                final = _ti.TimeStepper(inst, controller=pi).advance(
                    _ti.BlackBoxRHS(_base_f), _U2, 0.0, t_end
                )
                _assert_cons(final.u, 2, label=label)
                return _base_err(final.u, t_end)

            assert _run(tf) < _run(tc), f"{label}: tighter tol didn't reduce error"

        return pytest.param(_chk, id=f"pi_work_precision/{label}")

    def _auto_disp(
        label: str,
        rhs: object,
        state: _ti.ODEState,
        dt: float,
        direct: TimeIntegrator,
        order: int,
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            auto = _ti.AutoIntegrator()
            exp = direct.step(rhs, state, dt)
            act = auto.step(rhs, state, dt)
            assert direct.order == order
            assert act.t == exp.t and act.dt == exp.dt and act.err == exp.err
            assert (
                float(norm(act.u - exp.u)) < 1e-15
            ), f"{label}: AutoIntegrator mismatch"

        return pytest.param(_chk, id=f"auto_dispatch/{label}")

    def _type_coh(
        cls: type, insts: list[TimeIntegrator], label: str
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            assert issubclass(
                cls, TimeIntegrator
            ), f"{cls.__name__} must inherit TimeIntegrator"
            for inst in insts:
                assert isinstance(inst, TimeIntegrator), f"{inst!r} not TimeIntegrator"

        return pytest.param(_chk, id=f"type_coherence/{label}")

    def _newton(
        label: str,
        y_exp: list[float],
        gdt: float,
        f_vals: object,
        jac_vals: object,
        y_star: list[float],
        tol: float = 1e-11,
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            backend = Tensor([0.0]).backend
            n = len(y_exp)

            def f(y: Tensor) -> Tensor:
                return Tensor(f_vals([float(y[i]) for i in range(n)]), backend=backend)  # type: ignore[operator]

            def jac(y: Tensor) -> Tensor:
                vals = jac_vals([float(y[i]) for i in range(n)])  # type: ignore[operator]
                return Tensor(vals, backend=backend)

            y_sol = newton_solve(
                Tensor(y_exp, backend=backend), gamma_dt=gdt, f=f, jac=jac
            )
            for i, expected in enumerate(y_star):
                assert (
                    abs(float(y_sol[i]) - expected) < tol
                ), f"{label}: y[{i}]={float(y_sol[i]):.15f}"

        return pytest.param(_chk, id=f"newton_solve/{label}")

    def _proj_newt(
        label: str, k: float, gdt: float, y_exp: list[float]
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            cg = Tensor([[1.0, 1.0]])
            ye = Tensor(y_exp)

            def f(y: Tensor) -> Tensor:
                return Tensor([-k * float(y[0]), k * float(y[0])], backend=y.backend)

            def jac(y: Tensor) -> Tensor:
                return Tensor([[-k, 0.0], [k, 0.0]], backend=y.backend)

            res = newton_solve(ye, gdt, f, jac, constraint_gradients=cg)
            assert (
                abs(float(res[0]) + float(res[1]) - 1.0) < 1e-12
            ), f"{label}: constraint"
            y0e = y_exp[0] / (1.0 + gdt * k)
            assert abs(float(res[0]) - y0e) < 1e-10
            assert abs(float(res[1]) - (1.0 - y0e)) < 1e-10

        return pytest.param(_chk, id=f"projected_newton/{label}")

    def _nord_ra(
        inst: object,
        label: str,
        rhs: object,
        qs: int = 4,
        qt: int = 2,
        hs: float = 0.1,
        ht: float = 0.025,
    ) -> pytest.param:  # type: ignore[type-arg]
        def _chk() -> None:
            z_terms = [Tensor([1.0, 0.0])]
            for j in range(1, qs + 1):
                d = (-_k) ** j
                s = (hs**j) / math.factorial(j)
                z_terms.append(Tensor([s * d, -s * d]))
            nh = (
                _ti.NordsieckHistory(hs, tuple(z_terms))
                .change_order(qt)
                .rescale_step(ht)
            )
            state = _ti.ODEState(0.0, nh.z[0], ht, 0.0, nh)
            for _ in range(round((1.0 - state.t) / ht)):
                state = inst.step(rhs, state, ht)  # type: ignore[union-attr]
            fresh = inst.init_state(rhs, 0.0, Tensor([1.0, 0.0]), ht)  # type: ignore[union-attr]
            for _ in range(round((1.0 - fresh.t) / ht)):
                fresh = inst.step(rhs, fresh, ht)  # type: ignore[union-attr]
            terr = _base_err(state.u, state.t)
            ferr = _base_err(fresh.u, fresh.t)
            assert (
                terr <= 2.0 * ferr
            ), f"{label}: transformed {terr:.3e}>2×fresh {ferr:.3e}"

        return pytest.param(_chk, id=f"nordsieck_rescaled_accuracy/{label}")

    # ── single-instance checks ────────────────────────────────────────────────

    # A⇌B reaction network data (equal forward/reverse, exact X_A=0.5+0.5·exp(-2t))
    _RN_K = 1.0
    _RN_S = Tensor([[-1.0], [1.0]])
    _RN_U0 = Tensor([1.0, 0.0])

    def _rn_rp(t: float, u: Tensor) -> Tensor:
        return Tensor([_RN_K * float(u[0])], backend=u.backend)

    def _rn_rm(t: float, u: Tensor) -> Tensor:
        return Tensor([_RN_K * float(u[1])], backend=u.backend)

    def _rn_exact(t: float) -> tuple[float, float]:
        xa = 0.5 + 0.5 * math.exp(-2.0 * _RN_K * t)
        return xa, 1.0 - xa

    # Constraint lifecycle A⇌B (k_f=1.0, k_r=0.5, eq at A=1/3)
    _LC_S = Tensor([[-1.0], [1.0]])
    _LC_U0 = Tensor([1.0, 0.0])
    _LC_KF, _LC_KR = 1.0, 0.5

    def _lc_rhs() -> _ti.ReactionNetworkRHS:
        return _ti.ReactionNetworkRHS(
            _LC_S,
            lambda t, u: Tensor([_LC_KF * float(u[0])], backend=u.backend),
            lambda t, u: Tensor([_LC_KR * float(u[1])], backend=u.backend),
            _LC_U0,
        )

    def _lc_ctrl(rhs: _ti.ReactionNetworkRHS) -> _ti.ConstraintAwareController:
        return _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=_ti.implicit_midpoint,
            inner=_ti.PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=0.05),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )

    # NSE A⇌B⇌C  (k=1, eq at 1/3 each)
    _NSE_K = 1.0
    _NSE_S = Tensor([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]])
    _NSE_U0 = Tensor([1.0, 0.0, 0.0])

    def _nse_rhs() -> _ti.ReactionNetworkRHS:
        return _ti.ReactionNetworkRHS(
            _NSE_S,
            lambda t, u: Tensor(
                [_NSE_K * float(u[0]), _NSE_K * float(u[1])], backend=u.backend
            ),
            lambda t, u: Tensor(
                [_NSE_K * float(u[1]), _NSE_K * float(u[2])], backend=u.backend
            ),
            _NSE_U0,
        )

    # 10-species chain for rate-threshold guard
    _RT_ROWS = [[0.0] * 9 for _ in range(10)]
    for _j in range(9):
        _RT_ROWS[_j][_j] = -1.0
        _RT_ROWS[_j + 1][_j] = 1.0
    _RT_S = Tensor(_RT_ROWS)
    _RT_U0 = Tensor([1.0] + [0.0] * 9)
    _RT_K = 1.0

    def _rt_rp(t: float, u: Tensor) -> Tensor:
        return Tensor([_RT_K * float(u[j]) for j in range(9)], backend=u.backend)

    def _rt_rm(t: float, u: Tensor) -> Tensor:
        return Tensor([_RT_K * float(u[j + 1]) for j in range(9)], backend=u.backend)

    # Conservation projection data
    _PROJ_B = Tensor([[1.0, 1.0, 1.0]])
    _PROJ_T = Tensor([1.0])

    # ── single-instance check functions ──────────────────────────────────────

    def _chk_nord_rt() -> None:
        nh = _ti.NordsieckHistory(
            h=0.1,
            z=(Tensor([0.8, 0.2]), Tensor([-0.08, 0.08]), Tensor([0.004, -0.004])),
        )
        raised = nh.change_order(4)
        lowered = raised.change_order(2)
        assert lowered.q == nh.q and lowered.h == nh.h
        for lhs, rhs_ in zip(lowered.z, nh.z, strict=True):
            assert float(norm(lhs - rhs_)) == 0.0
        assert raised.q == 4
        assert float(norm(raised.z[0] - nh.z[0])) == 0.0
        assert float(norm(raised.z[3])) == 0.0 and float(norm(raised.z[4])) == 0.0
        rescaled = nh.rescale_step(0.05).rescale_step(0.1)
        assert rescaled.h == nh.h
        for lhs, rhs_ in zip(rescaled.z, nh.z, strict=True):
            assert float(norm(lhs - rhs_)) < 1e-16

    def _chk_var_order_climb() -> None:
        sel = _ti.OrderSelector(q_min=2, q_max=4, atol=1e-4, rtol=1e-4)
        inst = _ti.VariableOrderNordsieckIntegrator(_ti.adams_family, sel)
        state = inst.advance(
            _ti.BlackBoxRHS(_decay_f), _U3, t0=0.0, t_end=1.5, dt0=0.025
        )
        assert max(inst.accepted_orders) == 4 and inst.accepted_orders[-1] == 4
        x0, x1, x2 = _decay_exact(state.t)
        err = max(abs(float(state.u[i]) - v) for i, v in enumerate((x0, x1, x2)))
        assert err < 5e-4
        _assert_cons(state.u, 3, label="variable_order/climb")

    def _chk_var_order_drop() -> None:
        sel = _ti.OrderSelector(q_min=2, q_max=4, atol=1e-4, rtol=1e-4)
        inst = _ti.VariableOrderNordsieckIntegrator(_ti.adams_family, sel, q_initial=4)
        state = inst.advance(
            _ti.BlackBoxRHS(_sharpening_f), _U3, t0=0.0, t_end=1.0, dt0=0.02
        )
        post = [
            q
            for t, q in zip(inst.accepted_times, inst.accepted_orders, strict=True)
            if t > 0.55
        ]
        assert post and min(post) == 2
        _assert_cons(state.u, 3, label="variable_order/drop")
        for i in range(3):
            assert float(state.u[i]) >= -1e-10

    def _chk_stiff_diag() -> None:
        diag = _ti.StiffnessDiagnostic()
        jac = Tensor([[-3.0, 1.0], [0.5, -2.0]])
        below = diag.update(jac, h=0.1)
        above = diag.update(jac, h=0.3)
        assert abs(below - 0.4) < 1e-15 and abs(above - 1.2) < 1e-15
        assert below < 1.0 < above

    def _chk_fam_switch_rt() -> None:
        inst = _ti.FamilySwitchingNordsieckIntegrator(
            adams_family=_ti.adams_family,
            bdf_family=_ti.bdf_family,
            switcher=_ti.StiffnessSwitcher(),
            q=2,
            initial_family="adams",
        )
        z = (
            Tensor([0.8, 0.15, 0.05]),
            Tensor([-0.02, 0.01, 0.01]),
            Tensor([0.0, 0.0, 0.0]),
        )
        state = _ti.ODEState(0.25, z[0], 0.05, 0.0, _ti.NordsieckHistory(0.05, z))
        bdf_state = inst.transform_family(state, "bdf")
        rt = inst.transform_family(bdf_state, "adams")
        assert _ti.nordsieck_solution_distance(state, rt) == 0.0
        assert rt.history.h == state.history.h and rt.history.q == state.history.q

    def _chk_stiffening_switch() -> None:
        inst = _ti.FamilySwitchingNordsieckIntegrator(
            adams_family=_ti.adams_family,
            bdf_family=_ti.bdf_family,
            switcher=_ti.StiffnessSwitcher(stiff_threshold=1.0, nonstiff_threshold=0.4),
            q=2,
            initial_family="adams",
        )
        state = inst.advance(STIFFENING_RHS, _U3, t0=0.0, t_end=0.8, dt=0.02)
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
        _assert_cons(state.u, 3, label="stiffness/switch")
        for i in range(3):
            assert float(state.u[i]) >= -1e-10

    def _chk_vode() -> None:
        controller = _ti.VODEController(
            adams_family=_ti.adams_family,
            bdf_family=_ti.bdf_family,
            order_selector=_ti.OrderSelector(
                q_min=2, q_max=4, atol=5e-4, rtol=5e-4, factor_min=0.25, factor_max=1.15
            ),
            stiffness_switcher=_ti.StiffnessSwitcher(
                stiff_threshold=1.0, nonstiff_threshold=0.4
            ),
            q_initial=2,
            initial_family="adams",
        )
        state = controller.advance(VODE_RHS, _U3, t0=0.0, t_end=0.7, dt0=0.005)
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
        _assert_cons(state.u, 3, label="vode/switch")
        for i in range(3):
            assert float(state.u[i]) >= -1e-10

    def _chk_phi() -> None:
        A = Tensor([[0.0, 1.0], [0.0, 0.0]])
        v = Tensor([0.0, 1.0])
        expected = [
            Tensor([1.0, 1.0]),
            Tensor([0.5, 1.0]),
            Tensor([1.0 / 6.0, 0.5]),
            Tensor([1.0 / 24.0, 1.0 / 6.0]),
        ]
        for k, exp in enumerate(expected):
            phi = _ti.PhiFunction(k).apply(A, v)
            assert float(norm(phi - exp)) < 1e-14, f"phi_{k}: mismatch"

    def _chk_net_inv() -> None:
        rhs = _ti.ReactionNetworkRHS(
            stoichiometry_matrix=_RN_S,
            forward_rate=_rn_rp,
            reverse_rate=_rn_rm,
            initial_state=_RN_U0,
        )
        for i in range(2):
            assert float(rhs.stoichiometry_matrix[i, 0]) == pytest.approx(
                float(_RN_S[i, 0])
            )
        n_con, n_sp = rhs.conservation_basis.shape
        assert n_con == 1 and n_sp == 2
        row = [float(rhs.conservation_basis[0, j]) for j in range(2)]
        ratio = row[0] / row[1] if abs(row[1]) > 1e-14 else None
        assert ratio is not None and abs(ratio - 1.0) < 1e-12
        target = float(rhs.conservation_targets[0])
        expected = row[0] * float(_RN_U0[0]) + row[1] * float(_RN_U0[1])
        assert abs(target - expected) < 1e-12
        u_test = Tensor([0.7, 0.3])
        f_actual = rhs(0.0, u_test)
        r_net = _RN_K * (float(u_test[0]) - float(u_test[1]))
        assert float(f_actual[0]) == pytest.approx(-r_net, abs=1e-12)
        assert float(f_actual[1]) == pytest.approx(r_net, abs=1e-12)
        f_eq = rhs(0.0, Tensor([0.5, 0.5]))
        assert abs(float(f_eq[0])) < 1e-14 and abs(float(f_eq[1])) < 1e-14

    def _chk_proj_cons() -> None:
        u_off = Tensor([0.4, 0.4, 0.4])
        pu = _ti.project_conserved(u_off, _PROJ_B, _PROJ_T)
        ppu = _ti.project_conserved(pu, _PROJ_B, _PROJ_T)
        for i in range(3):
            assert abs(float(ppu[i]) - float(pu[i])) < 1e-12
        u_on = Tensor([0.5, 0.3, 0.2])
        pu_on = _ti.project_conserved(u_on, _PROJ_B, _PROJ_T)
        for i in range(3):
            assert abs(float(pu_on[i]) - float(u_on[i])) < 1e-12
        u_far = Tensor([0.5, 0.5, 0.5])
        pu_far = _ti.project_conserved(u_far, _PROJ_B, _PROJ_T)
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

    def _chk_activation() -> None:
        rhs = _lc_rhs()
        ctrl = _lc_ctrl(rhs)
        state = ctrl.advance(_LC_U0, 0.0, 6.0)
        assert state.active_constraints == frozenset({0})
        assert ctrl.activation_events

    def _chk_no_chatter() -> None:
        rhs = _lc_rhs()
        ctrl = _lc_ctrl(rhs)
        ctrl.advance(_LC_U0, 0.0, 6.0)
        assert not ctrl.deactivation_events, f"chattering: {ctrl.deactivation_events}"

    def _chk_consistent_init() -> None:
        rhs = _lc_rhs()
        ctrl = _lc_ctrl(rhs)
        state = ctrl.advance(_LC_U0, 0.0, 6.0)
        rp = float(rhs.forward_rate(state.t, state.u)[0])
        rm = float(rhs.reverse_rate(state.t, state.u)[0])
        ratio = abs(rp - rm) / max(abs(rp), abs(rm), 1e-100)
        assert ratio < 0.01, f"final ratio {ratio:.3e} >= eps_activate"

    def _chk_deactivation() -> None:
        rhs = _lc_rhs()
        ctrl = _lc_ctrl(rhs)
        state = ctrl.advance(
            Tensor([0.7, 0.3]), 0.0, 0.5, initial_active=frozenset({0})
        )
        assert ctrl.deactivation_events
        assert state.active_constraints == frozenset()

    def _chk_nse_direct() -> None:
        rhs = _nse_rhs()
        result = _ti.solve_nse(rhs, Tensor([0.34, 0.33, 0.33]), t=0.0)
        for i in range(3):
            xi = float(result[i])
            assert abs(xi - 1.0 / 3.0) < 1e-10, f"species {i}: {abs(xi - 1/3):.3e}"

    def _chk_nonlin_scalar() -> None:
        def F(x: Tensor) -> Tensor:
            return Tensor([float(x[0]) ** 2 - 2.0], backend=x.backend)

        def J(x: Tensor) -> Tensor:
            return Tensor([[2.0 * float(x[0])]], backend=x.backend)

        root = _ti.nonlinear_solve(F, J, Tensor([1.0]))
        assert abs(float(root[0]) - 2.0**0.5) < 1e-12, f"root={float(root[0]):.15f}"

    def _chk_rate_thresh() -> None:
        rhs = _ti.ReactionNetworkRHS(_RT_S, _rt_rp, _rt_rm, _RT_U0)
        ctrl = _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=_ti.implicit_midpoint,
            inner=_ti.PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=0.01),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        ctrl.advance(_RT_U0, 0.0, 0.05)
        for t_ev, pairs in ctrl.activation_events:
            for j in pairs:
                assert (
                    j == 0 or t_ev > 0.04
                ), f"spurious early activation pair {j} at t={t_ev:.4f}"

    # ── build the complete params list ────────────────────────────────────────

    _ORD_METHODS = [
        ("forward_euler", _ti.forward_euler),
        ("midpoint", _ti.midpoint),
        ("heun", _ti.heun),
        ("ralston", _ti.ralston),
        ("rk4", _ti.rk4),
        ("dormand_prince", _ti.dormand_prince),
        ("bogacki_shampine", _ti.bogacki_shampine),
        ("backward_euler", _ti.backward_euler),
        ("implicit_midpoint", _ti.implicit_midpoint),
        ("crouzeix_3", _ti.crouzeix_3),
    ]

    _cal = calibrate_cost()
    _CI_SPECS = _chain_specs(_CHAIN_N) + _spoke_specs(_SPOKE_N, _SPOKE_K)

    return [
        # ── RK order conditions ──────────────────────────────────────────────
        *[_rk_ord(n, m) for n, m in _ORD_METHODS],
        # ── convergence: explicit RK on 3-species decay chain ───────────────
        *[
            _cvg(
                f"rk/{n}",
                lambda dt, m=m: _decay_err(
                    _run_rk(m, _ti.BlackBoxRHS(_decay_f), _U3, dt).u, 1.0
                ),
                m.order,
            )
            for n, m in [
                ("forward_euler", _ti.forward_euler),
                ("midpoint", _ti.midpoint),
                ("heun", _ti.heun),
                ("ralston", _ti.ralston),
                ("rk4", _ti.rk4),
                ("dormand_prince", _ti.dormand_prince),
                ("bogacki_shampine", _ti.bogacki_shampine),
            ]
        ],
        # ── convergence: DIRK on 2-species base network ──────────────────────
        *[
            _cvg(
                f"dirk/{n}",
                lambda dt, m=m: _base_err(_run_rk(m, BASE_RHS, _U2, dt).u, 1.0),
                m.order,
            )
            for n, m in [
                ("backward_euler", _ti.backward_euler),
                ("implicit_midpoint", _ti.implicit_midpoint),
                ("crouzeix_3", _ti.crouzeix_3),
            ]
        ],
        # ── convergence: IMEX ────────────────────────────────────────────────
        _cvg(
            "imex/ars222",
            lambda dt: _base_err(_run_rk(_ti.ars222, BASE_IMEX, _U2, dt).u, 1.0),
            _ti.ars222.order,
        ),
        # ── convergence: Adams-Bashforth ─────────────────────────────────────
        *[
            _cvg(
                f"ab/{n}",
                lambda dt, m=m: _base_err(
                    _run_rk(m, _ti.BlackBoxRHS(_base_f), _U2, dt).u, 1.0
                ),
                m.order,
            )
            for n, m in [("ab2", _ti.ab2), ("ab3", _ti.ab3), ("ab4", _ti.ab4)]
        ],
        # ── convergence: BDF ─────────────────────────────────────────────────
        *[
            _cvg(
                f"bdf/{n}",
                lambda dt, m=m: (lambda s: _base_err(s.u, s.t))(
                    _run_ms(m, BASE_RHS, _U2, dt)
                ),
                m.order,
                dt_base=_DT / m.order,
                n_halvings=_BDF_NH - (math.floor(math.log2(m.order)) + 1),
            )
            for n, m in [
                ("bdf1", _ti.bdf1),
                ("bdf2", _ti.bdf2),
                ("bdf3", _ti.bdf3),
                ("bdf4", _ti.bdf4),
            ]
        ],
        # ── convergence: Adams-Moulton ───────────────────────────────────────
        *[
            _cvg(
                f"am/{n}",
                lambda dt, m=m: (lambda s: _base_err(s.u, s.t))(
                    _run_ms(m, _ti.BlackBoxRHS(_base_f), _U2, dt)
                ),
                m.order,
                dt_base=_DT / m.order,
                n_halvings=_BDF_NH - (math.floor(math.log2(m.order)) + 1),
            )
            for n, m in [
                ("am1", _ti.adams_moulton1),
                ("am2", _ti.adams_moulton2),
                ("am3", _ti.adams_moulton3),
                ("am4", _ti.adams_moulton4),
            ]
        ],
        # ── convergence: ETD (reference-solution tolerance 0.25) ─────────────
        *[
            _cvg(
                f"etd/{n}",
                _etd_cvg_fn(m),
                o,
                dt_base=0.025,
                n_halvings=2,
                tol=0.25,
            )
            for n, m, o in [
                ("etd_euler", _ti.etd_euler, 1),
                ("etdrk2", _ti.etdrk2, 2),
                ("cox_matthews_etdrk4", _ti.cox_matthews_etdrk4, 4),
                ("krogstad_etdrk4", _ti.krogstad_etdrk4, 4),
            ]
        ],
        # ── convergence: operator splitting ──────────────────────────────────
        _cvg(
            "splitting/lie",
            lambda dt: float(
                norm(
                    _integrate_split(
                        _ti.CompositionIntegrator(
                            [_ti.rk4, _ti.rk4], _ti.lie_steps(), order=1
                        ),
                        dt,
                    ).u
                    - _split_exact(1.0)
                )
            ),
            1,
            dt_base=0.1,
            n_halvings=3,
            tol=0.25,
        ),
        _cvg(
            "splitting/strang",
            lambda dt: float(
                norm(
                    _integrate_split(
                        _ti.CompositionIntegrator(
                            [_ti.rk4, _ti.rk4], _ti.strang_steps(), order=2
                        ),
                        dt,
                    ).u
                    - _split_exact(1.0)
                )
            ),
            2,
            dt_base=0.1,
            n_halvings=3,
            tol=0.25,
        ),
        _cvg(
            "splitting/yoshida",
            lambda dt: float(
                norm(
                    _integrate_split(
                        _ti.CompositionIntegrator(
                            [_ti.forward_euler, _ti.forward_euler],
                            _ti.yoshida_steps(),
                            order=4,
                        ),
                        dt,
                    ).u
                    - _split_exact(1.0)
                )
            ),
            4,
            dt_base=0.1,
            n_halvings=3,
            tol=0.25,
        ),
        # ── convergence: symplectic ───────────────────────────────────────────
        *[
            _cvg(f"symplectic/{n}", lambda dt, m=m: _run_sym(m, dt), m.order)
            for n, m in [
                ("symplectic_euler", _ti.symplectic_euler),
                ("leapfrog", _ti.leapfrog),
                ("forest_ruth", _ti.forest_ruth),
            ]
        ],
        _cvg(
            "symplectic/yoshida_6",
            lambda dt: _run_sym(_ti.yoshida_6, dt),
            _ti.yoshida_6.order,
            dt_base=0.25,
        ),
        _cvg(
            "symplectic/yoshida_8",
            lambda dt: _run_sym(_ti.yoshida_8, dt),
            _ti.yoshida_8.order,
            dt_base=0.25,
            n_halvings=3,
        ),
        # ── convergence: reaction-network A⇌B ────────────────────────────────
        *[
            _cvg(
                f"reaction_network/{n}",
                lambda dt, m=m: max(
                    abs(
                        float(
                            _run_rk(
                                m,
                                _ti.ReactionNetworkRHS(_RN_S, _rn_rp, _rn_rm, _RN_U0),
                                _U2,
                                dt,
                            ).u[i]
                        )
                        - _rn_exact(1.0)[i]
                    )
                    for i in range(2)
                ),
                m.order,
            )
            for n, m in [
                ("forward_euler", _ti.forward_euler),
                ("heun", _ti.heun),
                ("rk4", _ti.rk4),
            ]
        ],
        # ── conservation ─────────────────────────────────────────────────────
        _cons(
            "rk/rk4",
            lambda dt, t: _run_cons(_ti.rk4, DECAY_RHS, _U3, dt, t),
            3,
            _decay_exact,
            tol=1e-4,
        ),
        *[
            _cons(
                f"dirk/{n}",
                lambda dt, t, m=m: _run_cons(m, DECAY_RHS, _U3, dt, t),
                3,
                _decay_exact,
                tol=acc,
            )
            for n, m, acc in [
                ("backward_euler", _ti.backward_euler, 2e-2),
                ("implicit_midpoint", _ti.implicit_midpoint, 1e-3),
                ("crouzeix_3", _ti.crouzeix_3, 5e-4),
            ]
        ],
        _cons(
            "imex/ars222",
            lambda dt, t: _run_cons(_ti.ars222, DECAY_IMEX, _U3, dt, t),
            3,
            _decay_exact,
        ),
        *[
            _cons(
                f"ab/{n}",
                lambda dt, t, m=m: _run_cons(m, DECAY_RHS, _U3, dt, t),
                3,
                _decay_exact,
                tol=acc,
            )
            for n, m, acc in [
                ("ab2", _ti.ab2, 5e-4),
                ("ab3", _ti.ab3, 5e-5),
                ("ab4", _ti.ab4, 5e-6),
            ]
        ],
        *[
            _cons(
                f"bdf/{n}",
                lambda dt, t, m=m: _run_ms_cons(m, DECAY_RHS, _U3, dt, t),
                3,
                _decay_exact,
                tol=acc,
            )
            for n, m, acc in [
                ("bdf1", _ti.bdf1, 0.1),
                ("bdf2", _ti.bdf2, 5e-3),
                ("bdf3", _ti.bdf3, 5e-4),
                ("bdf4", _ti.bdf4, 5e-5),
            ]
        ],
        *[
            _cons(
                f"am/{n}",
                lambda dt, t, m=m: _run_ms_cons(m, DECAY_RHS, _U3, dt, t),
                3,
                _decay_exact,
                tol=acc,
            )
            for n, m, acc in [
                ("am1", _ti.adams_moulton1, 0.05),
                ("am2", _ti.adams_moulton2, 5e-4),
                ("am3", _ti.adams_moulton3, 1e-5),
                ("am4", _ti.adams_moulton4, 1e-6),
            ]
        ],
        # ── stepper, PI accuracy, PI work precision ───────────────────────────
        _stepper(_ti.forward_euler, "forward_euler", dt=1e-3, rtol=1e-3),
        _stepper(_ti.rk4, "rk4", dt=1e-2),
        _stepper(_ti.dormand_prince, "dormand_prince", dt=1e-2),
        *[
            _pi_acc(m, n)
            for m, n in [
                (_ti.heun, "heun"),
                (_ti.bogacki_shampine, "bogacki_shampine"),
                (_ti.dormand_prince, "dormand_prince"),
            ]
        ],
        *[
            _pi_wp(m, n)
            for m, n in [
                (_ti.heun, "heun"),
                (_ti.bogacki_shampine, "bogacki_shampine"),
                (_ti.dormand_prince, "dormand_prince"),
            ]
        ],
        # ── stability ────────────────────────────────────────────────────────
        *[
            _stab_a(n, m)
            for n, m in [
                ("backward_euler", _ti.backward_euler),
                ("implicit_midpoint", _ti.implicit_midpoint),
                ("crouzeix_3", _ti.crouzeix_3),
            ]
        ],
        _stab_l("backward_euler", _ti.backward_euler),
        # ── Nordsieck round trip ──────────────────────────────────────────────
        pytest.param(_chk_nord_rt, id="nordsieck_round_trip/order_and_step_size"),
        _nord_ra(_ti.bdf2, "bdf2", BASE_RHS),
        _nord_ra(_ti.adams_moulton2, "am2", _ti.BlackBoxRHS(_base_f)),
        pytest.param(
            _chk_var_order_climb, id="variable_order/climbs_on_smooth_network"
        ),
        pytest.param(
            _chk_var_order_drop, id="variable_order/drops_on_sharpening_network"
        ),
        pytest.param(_chk_stiff_diag, id="stiffness/diagnostic_threshold"),
        pytest.param(_chk_fam_switch_rt, id="stiffness/family_switch_round_trip"),
        pytest.param(
            _chk_stiffening_switch, id="stiffness/switcher_fires_on_stiffening_network"
        ),
        pytest.param(_chk_vode, id="vode_controller/fast_slow_family_policy"),
        pytest.param(_chk_phi, id="exponential/phi_function_coefficients"),
        _auto_disp(
            "explicit",
            _ti.BlackBoxRHS(_base_f),
            _ti.ODEState(0.0, _U2),
            0.05,
            _ti.rk4,
            4,
        ),
        _auto_disp(
            "implicit",
            DECAY_RHS,
            _ti.ODEState(0.0, _U3),
            0.05,
            _ti.implicit_midpoint,
            2,
        ),
        _auto_disp("split", DECAY_IMEX, _ti.ODEState(0.0, _U3), 0.05, _ti.ars222, 2),
        _auto_disp(
            "semilinear",
            ETD_RHS,
            _ti.ODEState(0.0, _U3),
            0.05,
            _ti.cox_matthews_etdrk4,
            4,
        ),
        _auto_disp(
            "composite",
            SPLIT_RHS,
            _ti.ODEState(0.0, _U2),
            0.05,
            _ti.CompositionIntegrator([_ti.rk4, _ti.rk4], _ti.strang_steps(), order=2),
            2,
        ),
        _auto_disp(
            "symplectic",
            _ti.HamiltonianRHS(dT_dp=lambda p: p, dV_dq=lambda q: q, split_index=1),
            _ti.ODEState(0.0, _U2),
            0.05,
            _ti.leapfrog,
            2,
        ),
        _type_coh(
            _ti.ImplicitRungeKuttaIntegrator,
            [_ti.backward_euler, _ti.implicit_midpoint, _ti.crouzeix_3],
            "ImplicitRungeKuttaIntegrator",
        ),
        _type_coh(
            _ti.AdditiveRungeKuttaIntegrator,
            [_ti.ars222],
            "AdditiveRungeKuttaIntegrator",
        ),
        _type_coh(
            _ti.ExplicitMultistepIntegrator,
            [_ti.ab2, _ti.ab3, _ti.ab4],
            "ExplicitMultistepIntegrator",
        ),
        _type_coh(
            _ti.SymplecticCompositionIntegrator,
            [
                _ti.symplectic_euler,
                _ti.leapfrog,
                _ti.forest_ruth,
                _ti.yoshida_6,
                _ti.yoshida_8,
            ],
            "SymplecticCompositionIntegrator",
        ),
        _type_coh(
            _ti.CompositionIntegrator,
            [
                _ti.CompositionIntegrator(
                    [_ti.rk4, _ti.rk4], _ti.strang_steps(), order=2
                )
            ],
            "CompositionIntegrator",
        ),
        _newton(
            "nonlinear_scalar",
            [0.9],
            0.1,
            lambda v: [v[0] * v[0]],
            lambda v: [[2.0 * v[0]]],
            [1.0],
        ),
        _newton(
            "linear_system",
            [1.0, 2.0],
            0.25,
            lambda v: [v[0], 2.0 * v[1]],
            lambda v: [[1.0, 0.0], [0.0, 2.0]],
            [4.0 / 3.0, 4.0],
        ),
        pytest.param(_chk_net_inv, id="reaction_network/invariants"),
        pytest.param(_chk_proj_cons, id="project_conserved/invariants"),
        _proj_newt("decay_k1_gh01", k=1.0, gdt=0.1, y_exp=[0.9, 0.1]),
        _proj_newt("decay_k5_gh02", k=5.0, gdt=0.2, y_exp=[0.7, 0.3]),
        pytest.param(_chk_activation, id="constraint_lifecycle/activation"),
        pytest.param(_chk_no_chatter, id="constraint_lifecycle/no_chattering"),
        pytest.param(_chk_consistent_init, id="constraint_lifecycle/consistent_init"),
        pytest.param(_chk_deactivation, id="constraint_lifecycle/deactivation"),
        pytest.param(_chk_nse_direct, id="nse_solver/direct_solve"),
        pytest.param(_chk_nonlin_scalar, id="nse_solver/nonlinear_solve_scalar"),
        pytest.param(_chk_rate_thresh, id="rate_threshold/absent_pairs"),
        *[_nse_claim(s) for s in _CI_SPECS],
        *[_cost_claim(s, _cal) for s in _CI_SPECS],
    ]


_ALL_PARAMS = _build_params()

_OFFLINE_PARAMS = [
    _nse_claim(s)
    for s in _chain_specs(_CHAIN_N_OFF) + _spoke_specs(_SPOKE_N_OFF, _SPOKE_K_OFF)
]

_OFFLINE_SKIP_REASON = (
    "offline network stress tests; "
    "set COSMIC_FOUNDRY_OFFLINE_NETWORK_STRESS=1 to run"
)


@pytest.mark.parametrize("check", _ALL_PARAMS)
def test_all(check: object) -> None:
    check()  # type: ignore[operator]


@pytest.mark.parametrize("check", _OFFLINE_PARAMS)
@pytest.mark.skipif(not _OFFLINE, reason=_OFFLINE_SKIP_REASON)
@pytest.mark.offline
def test_offline_network(check: object) -> None:
    check()  # type: ignore[operator]
