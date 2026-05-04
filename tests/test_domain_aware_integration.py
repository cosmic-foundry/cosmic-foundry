"""Domain-aware integration calculation claims."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.algorithm_capabilities import (
    MapStructureField,
    ParameterDescriptor,
    map_structure_parameter_schema,
)
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.capabilities import (
    rhs_step_diagnostics_descriptor,
)
from tests.claims import Claim

_TIME_BACKEND = NumpyBackend()


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


def _assert_abundance_state(u: Tensor, *, label: str) -> None:
    total = sum(float(u[i]) for i in range(u.shape[0]))
    assert abs(total - 1.0) < 1e-10, f"{label}: mass drift {total - 1.0:.3e}"
    minimum = min(float(u[i]) for i in range(u.shape[0]))
    assert minimum >= -1e-8, f"{label}: minimum abundance {minimum:.3e}"


def _two_species_decay_rhs(rate: float) -> _ti.ReactionNetworkRHS:
    return _ti.ReactionNetworkRHS(
        Tensor([[-1.0], [1.0]], backend=_TIME_BACKEND),
        lambda t, u: Tensor([rate * float(u[0])], backend=u.backend),
        lambda t, u: Tensor([0.0], backend=u.backend),
        Tensor([1.0, 0.0], backend=_TIME_BACKEND),
        jac=lambda t, u: Tensor([[-rate, 0.0], [rate, 0.0]], backend=u.backend),
    )


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


def _state_domain_claims() -> list[_DomainClaim]:
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
    ]


class _AdaptiveNordsieckDomainLimitClaim(Claim[Any]):
    @property
    def description(self) -> str:
        return "correctness/domain/adaptive_nordsieck_limits_negative_abundance"

    def check(self, _calibration: Any) -> None:
        rhs = _two_species_decay_rhs(300.0)
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


class _IntegrationDriverDomainLimitClaim(Claim[Any]):
    @property
    def description(self) -> str:
        return "correctness/domain/generic_integrator_limits_negative_abundance"

    def check(self, _calibration: Any) -> None:
        rhs = _two_species_decay_rhs(300.0)
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


class _ReactionNetworkStepDescriptorDomainLimitClaim(Claim[Any]):
    @property
    def description(self) -> str:
        return "correctness/domain/reaction_network_step_descriptor_limit"

    def check(self, _calibration: Any) -> None:
        rhs = _two_species_decay_rhs(300.0)
        state = _ti.ODEState(
            0.0,
            Tensor([1.0e-3, 1.0 - 1.0e-3], backend=_TIME_BACKEND),
        )
        dt = 5.0e-3
        descriptor = ParameterDescriptor(
            rhs.map_structure_descriptor().coordinates
            | rhs_step_diagnostics_descriptor(
                rhs,
                state,
                dt,
                local_error_target=1.0e-6,
                retry_budget=5,
            ).coordinates
        )
        schema = map_structure_parameter_schema()
        regions = {region.name: region for region in schema.derived_regions}
        schema.validate_descriptor(descriptor)

        limit = _ti.predict_domain_step_limit(rhs, state.t, state.u)
        assert limit is not None
        expected_margin = limit / dt - 1.0
        assert expected_margin < 0.0
        assert descriptor.coordinate(
            MapStructureField.DOMAIN_STEP_MARGIN
        ).value == pytest.approx(expected_margin)
        assert regions["domain_limited_step"].contains(descriptor)
        assert descriptor.coordinate(MapStructureField.LOCAL_ERROR_TARGET).value == (
            pytest.approx(1.0e-6)
        )
        assert descriptor.coordinate(MapStructureField.RETRY_BUDGET).value == 5


class _ConstraintAwareDomainLimitClaim(Claim[Any]):
    @property
    def description(self) -> str:
        return "correctness/domain/constraint_aware_limits_negative_abundance"

    def check(self, _calibration: Any) -> None:
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


_CORRECT_CLAIMS: tuple[Claim[Any], ...] = (
    *_state_domain_claims(),
    _ReactionNetworkStepDescriptorDomainLimitClaim(),
    _AdaptiveNordsieckDomainLimitClaim(),
    _IntegrationDriverDomainLimitClaim(),
    _ConstraintAwareDomainLimitClaim(),
)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[Any]) -> None:
    claim.check(None)
