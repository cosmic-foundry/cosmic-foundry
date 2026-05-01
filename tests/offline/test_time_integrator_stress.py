"""Offline stress tests for thermonuclear-shaped ODE integration.

These tests are intentionally punishing and opt-in.  They exercise synthetic
abundance networks with O(10) species, sparse zero-column-sum Jacobians,
large rate contrasts, positivity constraints, and abrupt regime changes.

Run explicitly with:

    COSMIC_FOUNDRY_OFFLINE_INTEGRATOR_STRESS=1 pytest tests/offline
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import pytest

from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators import (
    JacobianRHS,
    OrderSelector,
    StiffnessSwitcher,
    VODEController,
)

_RUN_OFFLINE = os.environ.get("COSMIC_FOUNDRY_OFFLINE_INTEGRATOR_STRESS") == "1"

pytestmark = [
    pytest.mark.offline,
    pytest.mark.skipif(
        not _RUN_OFFLINE,
        reason=(
            "offline integrator stress tests; set "
            "COSMIC_FOUNDRY_OFFLINE_INTEGRATOR_STRESS=1 to run"
        ),
    ),
]


RateFn = Callable[[float], list[tuple[int, int, float]]]


def _linear_network_rhs(rate_fn: RateFn) -> JacobianRHS:
    """Build a mass-conserving linear reaction network RHS from edge rates.

    Each edge ``src -> dst`` contributes ``-rate * X_src`` to the source and
    ``+rate * X_src`` to the destination, so every Jacobian column sums to
    zero exactly.
    """

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

    return JacobianRHS(f=f, jac=jac)


def _alpha_chain_rates(n: int) -> RateFn:
    """Return a stiff alpha-chain-like topology with rates spanning 8 decades."""
    rates = [10.0 ** (-2.0 + 8.0 * i / (n - 2)) for i in range(n - 1)]

    def edges(t: float) -> list[tuple[int, int, float]]:
        return [(i, i + 1, rates[i]) for i in range(n - 1)]

    return edges


def _branched_hot_window_rates(n: int) -> RateFn:
    """Return a branched network with an abrupt hot window and breakout path."""
    base_edges: list[tuple[int, int, float]] = []
    for i in range(n - 1):
        base_edges.append((i, i + 1, 0.04 * (1.25**i)))
    branches = [
        (2, 7, 0.03),
        (4, 10, 0.02),
        (6, 13, 0.015),
        (8, 15, 0.012),
    ]

    def edges(t: float) -> list[tuple[int, int, float]]:
        hot = 1.0 + 2.0e4 * math.exp(-(((t - 0.18) / 0.025) ** 2))
        breakout = 1.0 + 5.0e5 * math.exp(-(((t - 0.24) / 0.018) ** 2))
        result = [(src, dst, rate * hot) for src, dst, rate in base_edges]
        result.extend((src, dst, rate * breakout) for src, dst, rate in branches)
        return result

    return edges


def _vode_controller(*, dt0: float, q_max: int = 6) -> VODEController:
    return VODEController(
        order_selector=OrderSelector(
            q_min=2,
            q_max=q_max,
            atol=2e-5,
            rtol=2e-5,
            factor_min=0.2,
            factor_max=1.15,
        ),
        stiffness_switcher=StiffnessSwitcher(
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


@pytest.mark.timeout(20)
def test_offline_vode_alpha_chain_rate_contrast() -> None:
    """Stress VODE on a 13-species capture chain spanning eight rate decades."""
    rhs = _linear_network_rhs(_alpha_chain_rates(13))
    controller = _vode_controller(dt0=2e-4)
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


@pytest.mark.timeout(30)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "current VODE controller does not enforce positivity; this branched "
        "hot-window network exposes small negative abundances"
    ),
)
def test_offline_vode_branched_hot_window_self_consistency() -> None:
    """Stress VODE on a 16-species branched network with a transient hot window."""
    rhs = _linear_network_rhs(_branched_hot_window_rates(16))
    coarse = _vode_controller(dt0=5e-4)
    fine = _vode_controller(dt0=2.5e-4)

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
