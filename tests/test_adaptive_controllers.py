"""Adaptive-controller calculation claims."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import pytest

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.theory.discrete import FiniteStateTransitionSystem
from tests.claims import Claim

_TIME_BACKEND = NumpyBackend()

RateFn = Callable[[float], list[tuple[int, int, float]]]


def _finite_transition_initial_state(
    system: FiniteStateTransitionSystem,
) -> Tensor:
    return Tensor([1.0] + [0.0] * (system.state_count - 1), backend=_TIME_BACKEND)


def _unit_transfer_rhs(
    system: FiniteStateTransitionSystem,
    rates: _ti.UnitTransferRates,
) -> _ti.ReactionNetworkRHS:
    return _ti.ReactionNetworkRHS.from_unit_transfer_system(
        system,
        rates,
        _finite_transition_initial_state(system),
    )


def _linear_network_rhs(rate_fn: RateFn, n: int) -> _ti.ReactionNetworkRHS:
    edges0 = rate_fn(0.0)
    system = FiniteStateTransitionSystem(
        n,
        tuple((src, dst) for src, dst, _rate in edges0),
    )
    return _unit_transfer_rhs(
        system,
        lambda t: tuple(rate for _src, _dst, rate in rate_fn(t)),
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


def _adaptive_nordsieck_controller() -> _ti.AdaptiveNordsieckController:
    return _ti.AdaptiveNordsieckController(
        order_selector=_ti.OrderSelector(
            q_min=2,
            q_max=6,
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


class _AlphaChainStiffnessSwitchClaim(Claim[Any]):
    @property
    def description(self) -> str:
        return "correctness/stress/adaptive_nordsieck_alpha_chain_rate_contrast"

    @property
    def expected_walltime_s(self) -> float:
        return 20.0

    def check(self, _calibration: Any) -> None:
        self.skip_if_over_walltime_budget()
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


class _BranchedHotWindowStiffnessClaim(Claim[Any]):
    @property
    def description(self) -> str:
        return "correctness/stress/adaptive_nordsieck_branched_hot_window"

    @property
    def expected_walltime_s(self) -> float:
        return 30.0

    def check(self, _calibration: Any) -> None:
        self.skip_if_over_walltime_budget()
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


_CORRECT_CLAIMS: tuple[Claim[Any], ...] = (
    _AlphaChainStiffnessSwitchClaim(),
    _BranchedHotWindowStiffnessClaim(),
)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[Any]) -> None:
    claim.check(None)
