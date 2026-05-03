"""Algorithm-selection calculation claims."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmRequest,
    CoverageRegion,
    MapStructureField,
    ParameterDescriptor,
    SolveRelationField,
    map_structure_parameter_schema,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.capabilities import (
    composite_map_descriptor,
    composite_map_descriptor_from_rhs,
    derivative_oracle_descriptor,
    hamiltonian_map_descriptor,
    nordsieck_history_descriptor,
    rhs_evaluation_descriptor,
    rhs_history_descriptor,
    semilinear_map_descriptor,
    split_map_descriptor,
    time_integration_step_map_regions,
    time_integration_step_solve_regions,
)
from tests import time_integrator_cases as cases
from tests.claims import Claim

_TIME_BACKEND = cases.TIME_BACKEND
_DAMPED_OSCILLATOR_GAMMA = 0.2


def _owned_region_owner(
    descriptor: ParameterDescriptor,
    regions: tuple[CoverageRegion, ...],
) -> type:
    owners = tuple(region.owner for region in regions if region.contains(descriptor))
    assert len(owners) == 1
    return owners[0]


def _assert_owned_cell(
    descriptor: ParameterDescriptor,
    *,
    regions: tuple[CoverageRegion, ...],
    schema: Any,
) -> None:
    schema.validate_descriptor(descriptor)
    assert schema.cell_status(descriptor, regions) == "owned"
    assert _owned_region_owner(descriptor, regions)


def _exact_damped_osc(t: float) -> tuple[float, ...]:
    gamma = _DAMPED_OSCILLATOR_GAMMA
    omega = math.sqrt(1.0 - (0.5 * gamma) ** 2)
    envelope = math.exp(-0.5 * gamma * t)
    sin_term = math.sin(omega * t)
    cos_term = math.cos(omega * t)
    q = envelope * (cos_term + 0.5 * gamma / omega * sin_term)
    p = envelope * sin_term / omega
    return q, p


def _oscillator_energy(u: Tensor) -> float:
    return 0.5 * sum(float(u[i]) ** 2 for i in range(u.shape[0]))


def _uncertified_oscillator_composite_rhs() -> _ti.CompositeRHS:
    return _ti.CompositeRHS(
        [
            _ti.ComponentFlowRHS(
                lambda t, u: Tensor([-float(u[1]), 0.0], backend=u.backend),
                symplectic_form_defect_upper_bound=float("inf"),
            ),
            _ti.ComponentFlowRHS(
                lambda t, u: Tensor([0.0, float(u[0])], backend=u.backend),
                symplectic_form_defect_upper_bound=float("inf"),
            ),
        ]
    )


def _damped_oscillator_composite_rhs() -> _ti.CompositeRHS:
    gamma = _DAMPED_OSCILLATOR_GAMMA
    return _ti.CompositeRHS(
        [
            _ti.ComponentFlowRHS(
                lambda t, u: Tensor([-float(u[1]), 0.0], backend=u.backend),
                symplectic_form_defect_upper_bound=0.0,
            ),
            _ti.ComponentFlowRHS(
                lambda t, u: Tensor(
                    [0.0, float(u[0]) - gamma * float(u[1])],
                    backend=u.backend,
                ),
                symplectic_form_defect_upper_bound=math.sqrt(2.0) * gamma,
            ),
        ]
    )


def _auto_selection_rhs_by_owner() -> dict[type, object]:
    return {
        _ti.RungeKuttaIntegrator: _ti.BlackBoxRHS(
            lambda t, u: Tensor([-float(u[0]), float(u[0])], backend=u.backend)
        ),
        _ti.ImplicitRungeKuttaIntegrator: cases.scalar_decay_jacobian_rhs(),
        _ti.AdditiveRungeKuttaIntegrator: cases.split_decay_rhs(),
        _ti.LawsonRungeKuttaIntegrator: cases.semilinear_forcing_rhs(),
        _ti.CompositionIntegrator: cases.oscillator_composite_rhs(),
        _ti.SymplecticCompositionIntegrator: cases.harmonic_hamiltonian_rhs(),
    }


@dataclass(frozen=True)
class _StepSelectionCase:
    name: str
    descriptor: ParameterDescriptor
    order: int
    state: _ti.ODEState
    rhs: object
    integrator: object
    exact: Callable[[float], tuple[float, ...]]
    tolerance: float
    regions: tuple[CoverageRegion, ...]
    schema: Any
    postcheck: Callable[[_StepSelectionCase, _ti.ODEState], None] = (
        lambda case, state: None
    )

    @property
    def owner(self) -> type:
        return _owned_region_owner(self.descriptor, self.regions)

    def step(self, dt: float) -> _ti.ODEState:
        return self.integrator.step(self.rhs, self.state, dt)  # type: ignore[attr-defined]


def _map_selection_case(
    name: str,
    descriptor: ParameterDescriptor,
    order: int,
    rhs: object,
    integrator: object,
    u0: Tensor,
    exact: Callable[[float], tuple[float, ...]],
    tolerance: float,
    postcheck: Callable[[_StepSelectionCase, _ti.ODEState], None] = (
        lambda case, state: None
    ),
) -> _StepSelectionCase:
    return _StepSelectionCase(
        name=name,
        descriptor=descriptor,
        order=order,
        state=_ti.ODEState(0.0, u0),
        rhs=rhs,
        integrator=integrator,
        exact=exact,
        tolerance=tolerance,
        regions=time_integration_step_map_regions(),
        schema=map_structure_parameter_schema(),
        postcheck=postcheck,
    )


def _solve_selection_case() -> _StepSelectionCase:
    rhs = cases.scalar_decay_jacobian_rhs()
    integrator = _ti.ImplicitRungeKuttaIntegrator(2)
    state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
    dt = 1.0e-2
    descriptor = integrator.step_solve_relation_descriptor(rhs, state, dt)

    def postcheck(case: _StepSelectionCase, state: _ti.ODEState) -> None:
        assert case.descriptor.coordinate(
            SolveRelationField.DERIVATIVE_ORACLE_KIND
        ) == derivative_oracle_descriptor().coordinate(
            SolveRelationField.DERIVATIVE_ORACLE_KIND
        )

    return _StepSelectionCase(
        name="implicit_stage_solve",
        descriptor=descriptor,
        order=2,
        state=state,
        rhs=rhs,
        integrator=integrator,
        exact=cases.exact_scalar_decay,
        tolerance=1.0e-7,
        regions=time_integration_step_solve_regions(),
        schema=solve_relation_parameter_schema(),
        postcheck=postcheck,
    )


def _rhs_history_selection_case() -> _StepSelectionCase:
    rhs = cases.scalar_decay_jacobian_rhs()
    dt = 1.0e-2
    integrator = _ti.ExplicitMultistepIntegrator.for_order(4)
    state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
    for _ in range(3):
        state = integrator.step(rhs, state, dt)
    assert isinstance(state.history, tuple)
    assert len(state.history) == 3
    return _StepSelectionCase(
        name="rhs_history",
        descriptor=rhs_history_descriptor(),
        order=4,
        state=state,
        rhs=rhs,
        integrator=integrator,
        exact=cases.exact_scalar_decay,
        tolerance=1.0e-8,
        regions=time_integration_step_map_regions(),
        schema=map_structure_parameter_schema(),
    )


def _nordsieck_history_selection_case() -> _StepSelectionCase:
    rhs = cases.scalar_decay_jacobian_rhs()
    integrator = _ti.MultistepIntegrator("adams", 4)
    state = integrator.init_state(rhs, 0.0, Tensor([1.0], backend=_TIME_BACKEND), 1e-2)
    assert isinstance(state.history, _ti.NordsieckHistory)
    assert state.history.q == 4
    return _StepSelectionCase(
        name="nordsieck_history",
        descriptor=nordsieck_history_descriptor(),
        order=4,
        state=state,
        rhs=rhs,
        integrator=integrator,
        exact=cases.exact_scalar_decay,
        tolerance=1.0e-8,
        regions=time_integration_step_map_regions(),
        schema=map_structure_parameter_schema(),
    )


def _assert_no_step_solve_or_linear_operator(
    case: _StepSelectionCase,
    state: _ti.ODEState,
) -> None:
    for method in (
        "step_solve_relation_descriptor",
        "step_linear_operator_descriptor",
    ):
        candidate = getattr(case.integrator, method, None)
        if callable(candidate):
            with pytest.raises(ValueError):
                candidate(case.rhs, state, 1.0e-2)


def _composition_postcheck(case: _StepSelectionCase, state: _ti.ODEState) -> None:
    assert (
        case.descriptor.coordinate(MapStructureField.ADDITIVE_COMPONENT_COUNT).value
        == 2
    )
    assert not case.descriptor.coordinate(
        MapStructureField.HAMILTONIAN_PARTITION_AVAILABLE
    ).value
    assert case.descriptor.coordinate(
        MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
    ).value
    generic_descriptor = composite_map_descriptor(len(case.rhs.components))  # type: ignore[attr-defined]
    assert not generic_descriptor.coordinate(
        MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
    ).value
    _assert_no_step_solve_or_linear_operator(case, state)


def _step_selection_cases() -> tuple[_StepSelectionCase, ...]:
    return (
        _map_selection_case(
            "rhs_evaluation",
            rhs_evaluation_descriptor(),
            4,
            _ti.BlackBoxRHS(lambda t, u: Tensor([-float(u[0])], backend=u.backend)),
            _ti.RungeKuttaIntegrator(4),
            Tensor([1.0], backend=_TIME_BACKEND),
            cases.exact_scalar_decay,
            1.0e-10,
        ),
        _rhs_history_selection_case(),
        _nordsieck_history_selection_case(),
        _solve_selection_case(),
        _map_selection_case(
            "split_map",
            split_map_descriptor(),
            2,
            cases.split_decay_rhs(),
            _ti.AdditiveRungeKuttaIntegrator(2),
            Tensor([1.0], backend=_TIME_BACKEND),
            cases.exact_scalar_decay,
            1.0e-7,
            lambda case, state: _assert_no_step_solve_or_linear_operator(case, state),
        ),
        _map_selection_case(
            "semilinear_map",
            semilinear_map_descriptor(),
            4,
            cases.semilinear_forcing_rhs(),
            _ti.LawsonRungeKuttaIntegrator(4),
            Tensor([1.0], backend=_TIME_BACKEND),
            cases.exact_semilinear,
            1.0e-10,
            lambda case, state: _assert_no_step_solve_or_linear_operator(case, state),
        ),
        _map_selection_case(
            "hamiltonian_map",
            hamiltonian_map_descriptor(),
            4,
            cases.harmonic_hamiltonian_rhs(),
            _ti.SymplecticCompositionIntegrator(4),
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
            cases.exact_ham,
            1.0e-10,
            lambda case, state: _assert_no_step_solve_or_linear_operator(case, state),
        ),
        _map_selection_case(
            "composition_map",
            composite_map_descriptor_from_rhs(cases.oscillator_composite_rhs()),
            4,
            cases.oscillator_composite_rhs(),
            _ti.CompositionIntegrator(
                [_ti.RungeKuttaIntegrator(1), _ti.RungeKuttaIntegrator(1)],
                order=4,
            ),
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
            cases.exact_osc,
            1.0e-8,
            _composition_postcheck,
        ),
    )


class _StepSelectionClaim(Claim[Any]):
    """Grounded claim for descriptor-region ownership and selector agreement."""

    def __init__(self, case: _StepSelectionCase) -> None:
        self._case = case

    @property
    def description(self) -> str:
        return f"correctness/step_selection/{self._case.name}"

    def check(self, _calibration: Any) -> None:
        case = self._case
        _assert_owned_cell(
            case.descriptor,
            regions=case.regions,
            schema=case.schema,
        )
        selected = _ti.select_time_integrator(
            AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=case.order,
                descriptor=case.descriptor,
            )
        )
        assert selected.implementation == case.owner.__name__
        state = case.step(1.0e-2)
        assert cases.err(state.u, case.exact, state.t) < case.tolerance
        case.postcheck(case, state)


class _StepSelectionRegionCoverageClaim(Claim[Any]):
    """Claim that selection calculations cover every step ownership region."""

    @property
    def description(self) -> str:
        return "correctness/step_selection_region_coverage"

    def check(self, _calibration: Any) -> None:
        cases_by_region = _step_selection_cases()
        for region in (
            *time_integration_step_map_regions(),
            *time_integration_step_solve_regions(),
        ):
            assert any(
                region.owner is case.owner and region.contains(case.descriptor)
                for case in cases_by_region
            ), region.owner.__name__


class _OscillatorInvariantComparisonClaim(Claim[Any]):
    """Grounded claim that component count alone does not encode invariance."""

    @property
    def description(self) -> str:
        return "correctness/oscillator_invariant_comparison"

    def check(self, _calibration: Any) -> None:
        dt = 5.0e-2
        steps = 400
        initial = Tensor([1.0, 0.0], backend=_TIME_BACKEND)
        composition = _ti.ODEState(0.0, initial)
        hamiltonian = _ti.ODEState(0.0, initial)
        composition_integrator = _ti.CompositionIntegrator(
            [_ti.RungeKuttaIntegrator(1), _ti.RungeKuttaIntegrator(1)],
            order=4,
        )
        hamiltonian_integrator = _ti.SymplecticCompositionIntegrator(4)
        composition_rhs = cases.oscillator_composite_rhs()
        hamiltonian_rhs = cases.harmonic_hamiltonian_rhs()
        composition_descriptor = composite_map_descriptor_from_rhs(composition_rhs)
        hamiltonian_descriptor = hamiltonian_map_descriptor()

        assert composition_descriptor.coordinate(
            MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
        ).value
        assert hamiltonian_descriptor.coordinate(
            MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
        ).value
        assert (
            _ti.select_time_integrator(
                AlgorithmRequest(
                    requested_properties=frozenset({"one_step"}),
                    order=4,
                    descriptor=composition_descriptor,
                )
            ).implementation
            == _ti.CompositionIntegrator.__name__
        )
        assert (
            _ti.select_time_integrator(
                AlgorithmRequest(
                    requested_properties=frozenset({"one_step"}),
                    order=4,
                    descriptor=hamiltonian_descriptor,
                )
            ).implementation
            == _ti.SymplecticCompositionIntegrator.__name__
        )

        initial_energy = _oscillator_energy(initial)
        composition_peak = 0.0
        hamiltonian_peak = 0.0
        for _ in range(steps):
            composition = composition_integrator.step(composition_rhs, composition, dt)
            hamiltonian = hamiltonian_integrator.step(hamiltonian_rhs, hamiltonian, dt)
            composition_peak = max(
                composition_peak,
                abs(_oscillator_energy(composition.u) - initial_energy),
            )
            hamiltonian_peak = max(
                hamiltonian_peak,
                abs(_oscillator_energy(hamiltonian.u) - initial_energy),
            )

        assert hamiltonian_peak < 1.0e-5
        assert composition_peak < 1.0e-5


class _OscillatorNegativeInvariantEvidenceClaim(Claim[Any]):
    """Grounded claim that invariant evidence is not composition ownership."""

    @property
    def description(self) -> str:
        return "correctness/oscillator_invariant_evidence_overlay"

    def check(self, _calibration: Any) -> None:
        certified_rhs = cases.oscillator_composite_rhs()
        uncertified_rhs = _uncertified_oscillator_composite_rhs()
        certified_descriptor = composite_map_descriptor_from_rhs(certified_rhs)
        uncertified_descriptor = composite_map_descriptor_from_rhs(uncertified_rhs)
        integrator = _ti.CompositionIntegrator(
            [_ti.RungeKuttaIntegrator(1), _ti.RungeKuttaIntegrator(1)],
            order=4,
        )
        certified = _ti.ODEState(0.0, Tensor([1.0, 0.0], backend=_TIME_BACKEND))
        uncertified = _ti.ODEState(0.0, Tensor([1.0, 0.0], backend=_TIME_BACKEND))
        dt = 5.0e-2
        steps = 200

        assert (
            certified_descriptor.coordinate(
                MapStructureField.ADDITIVE_COMPONENT_COUNT
            ).value
            == 2
        )
        assert (
            uncertified_descriptor.coordinate(
                MapStructureField.ADDITIVE_COMPONENT_COUNT
            ).value
            == 2
        )
        assert certified_descriptor.coordinate(
            MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
        ).value
        assert (
            certified_descriptor.coordinate(
                MapStructureField.SYMPLECTIC_FORM_DEFECT_UPPER_BOUND
            ).value
            == 0.0
        )
        assert not uncertified_descriptor.coordinate(
            MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
        ).value
        assert math.isinf(
            uncertified_descriptor.coordinate(
                MapStructureField.SYMPLECTIC_FORM_DEFECT_UPPER_BOUND
            ).value
        )
        for descriptor in (certified_descriptor, uncertified_descriptor):
            owner = _owned_region_owner(descriptor, time_integration_step_map_regions())
            assert (
                _ti.select_time_integrator(
                    AlgorithmRequest(
                        requested_properties=frozenset({"one_step"}),
                        order=4,
                        descriptor=descriptor,
                    )
                ).implementation
                == owner.__name__
            )
            _assert_owned_cell(
                descriptor,
                regions=time_integration_step_map_regions(),
                schema=map_structure_parameter_schema(),
            )

        for _ in range(steps):
            certified = integrator.step(certified_rhs, certified, dt)
            uncertified = integrator.step(uncertified_rhs, uncertified, dt)

        assert cases.err(certified.u, cases.exact_osc, certified.t) < 1.0e-5
        assert cases.err(uncertified.u, cases.exact_osc, uncertified.t) < 1.0e-5
        assert (
            max(abs(float(certified.u[i]) - float(uncertified.u[i])) for i in range(2))
            < 1.0e-14
        )


class _DampedOscillatorSymplecticDefectClaim(Claim[Any]):
    """Grounded claim for positive symplectic-defect evidence."""

    @property
    def description(self) -> str:
        return "correctness/damped_oscillator_symplectic_defect"

    def check(self, _calibration: Any) -> None:
        rhs = _damped_oscillator_composite_rhs()
        descriptor = composite_map_descriptor_from_rhs(rhs)
        integrator = _ti.CompositionIntegrator(
            [_ti.RungeKuttaIntegrator(1), _ti.RungeKuttaIntegrator(1)],
            order=2,
        )
        state = _ti.ODEState(0.0, Tensor([1.0, 0.0], backend=_TIME_BACKEND))
        dt = 2.5e-3
        steps = 400

        assert (
            descriptor.coordinate(MapStructureField.ADDITIVE_COMPONENT_COUNT).value == 2
        )
        assert not descriptor.coordinate(
            MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
        ).value
        assert descriptor.coordinate(
            MapStructureField.SYMPLECTIC_FORM_DEFECT_UPPER_BOUND
        ).value == pytest.approx(math.sqrt(2.0) * _DAMPED_OSCILLATOR_GAMMA)
        assert (
            _ti.select_time_integrator(
                AlgorithmRequest(
                    requested_properties=frozenset({"one_step"}),
                    order=2,
                    descriptor=descriptor,
                )
            ).implementation
            == _owned_region_owner(
                descriptor, time_integration_step_map_regions()
            ).__name__
        )
        _assert_owned_cell(
            descriptor,
            regions=time_integration_step_map_regions(),
            schema=map_structure_parameter_schema(),
        )

        initial_energy = _oscillator_energy(state.u)
        for _ in range(steps):
            state = integrator.step(rhs, state, dt)

        assert cases.err(state.u, _exact_damped_osc, state.t) < 2.0e-3
        assert _oscillator_energy(state.u) < initial_energy


class _AutoIntegratorSelectionClaim(Claim[Any]):
    """Grounded claim for RHS-type projection into capability selection."""

    @property
    def description(self) -> str:
        return "correctness/auto_integrator_selects_capability_branches"

    def check(self, _calibration: Any) -> None:
        auto = _ti.AutoIntegrator(4)
        for owner, rhs in _auto_selection_rhs_by_owner().items():
            assert auto.select(rhs).implementation == owner.__name__

        with pytest.raises(ValueError, match="no algorithm"):
            _ti.AutoIntegrator(3).select(cases.harmonic_hamiltonian_rhs())


_CORRECT_CLAIMS: tuple[Claim[Any], ...] = (
    _AutoIntegratorSelectionClaim(),
    _StepSelectionRegionCoverageClaim(),
    *[_StepSelectionClaim(case) for case in _step_selection_cases()],
    _OscillatorInvariantComparisonClaim(),
    _OscillatorNegativeInvariantEvidenceClaim(),
    _DampedOscillatorSymplecticDefectClaim(),
)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[Any]) -> None:
    claim.check(None)
