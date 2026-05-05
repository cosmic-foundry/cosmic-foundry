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
    LinearSolverField,
    MapStructureField,
    ParameterDescriptor,
    SolveRelationField,
    assembled_linear_evidence_for,
    linear_operator_descriptor_from_solve_relation_descriptor,
    map_structure_parameter_schema,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.solvers import (
    DenseLUSolver,
    DirectionalDerivativeRootRelation,
    FiniteDimensionalResidualRelation,
    LinearResidualRelation,
    MatrixFreeNewtonKrylovRootSolver,
    NewtonRootSolver,
)
from cosmic_foundry.computation.solvers.capabilities import (
    select_linear_solver_for_descriptor,
    select_root_solver_for_descriptor,
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
    rhs_step_diagnostics_descriptor,
    semilinear_map_descriptor,
    split_map_descriptor,
    time_integration_step_map_regions,
    time_integration_step_solve_regions,
)
from tests import time_integrator_cases as cases
from tests.claims import Claim
from tests.selection_ownership import SelectionOwnership

_TIME_BACKEND = cases.TIME_BACKEND
_DAMPED_OSCILLATOR_GAMMA = 0.2


class _AffineDecayRHS:
    def __init__(self, rate: float = 1.0) -> None:
        self._rate = rate

    def __call__(self, t: float, u: Tensor) -> Tensor:
        return self.linear_operator(t, u) @ u

    def linear_operator(self, _t: float, u: Tensor) -> Tensor:
        return Tensor([[-self._rate]], backend=u.backend)

    def jacobian(self, _t: float, u: Tensor) -> Tensor:
        return self.linear_operator(_t, u)


class _BoundaryApproachRHS:
    state_domain = _ti.NonnegativeStateDomain(1)

    def __call__(self, _t: float, u: Tensor) -> Tensor:
        return Tensor([-1.0], backend=u.backend)

    def jacobian(self, _t: float, u: Tensor) -> Tensor:
        return Tensor([[0.0]], backend=u.backend)


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


@dataclass(frozen=True)
class _StepSelectionCase:
    name: str
    descriptor: ParameterDescriptor
    order: int
    state: _ti.ODEState
    rhs: object
    exact: Callable[[float], tuple[float, ...]]
    tolerance: float
    ownership: SelectionOwnership
    auto_selectable: bool = True
    postcheck: Callable[[_StepSelectionCase, _ti.ODEState, object], None] = (
        lambda case, state, integrator: None
    )

    @property
    def owner(self) -> type:
        return self.ownership.owner


def _map_selection_case(
    name: str,
    descriptor: ParameterDescriptor,
    order: int,
    rhs: object,
    u0: Tensor,
    exact: Callable[[float], tuple[float, ...]],
    tolerance: float,
    postcheck: Callable[[_StepSelectionCase, _ti.ODEState, object], None] = (
        lambda case, state, integrator: None
    ),
) -> _StepSelectionCase:
    return _StepSelectionCase(
        name=name,
        descriptor=descriptor,
        order=order,
        state=_ti.ODEState(0.0, u0),
        rhs=rhs,
        exact=exact,
        tolerance=tolerance,
        ownership=SelectionOwnership(
            descriptor,
            time_integration_step_map_regions(),
            map_structure_parameter_schema(),
        ),
        postcheck=postcheck,
    )


def _solve_selection_case() -> _StepSelectionCase:
    rhs = cases.scalar_decay_jacobian_rhs()
    descriptor_integrator = _ti.ImplicitRungeKuttaIntegrator(2)
    state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
    dt = 1.0e-2
    descriptor = descriptor_integrator.step_solve_relation_descriptor(rhs, state, dt)
    root_relation = _ti.implicit_stage_root_relation(
        descriptor_integrator,
        rhs,
        state,
        dt,
    )
    assert isinstance(root_relation, FiniteDimensionalResidualRelation)
    root_descriptor = root_relation.solve_relation_descriptor()

    def postcheck(
        case: _StepSelectionCase,
        state: _ti.ODEState,
        _integrator: object,
    ) -> None:
        for field in SolveRelationField:
            assert case.descriptor.coordinate(field) == root_descriptor.coordinate(
                field
            )
        assert case.descriptor.coordinate(
            SolveRelationField.DERIVATIVE_ORACLE_KIND
        ) == derivative_oracle_descriptor().coordinate(
            SolveRelationField.DERIVATIVE_ORACLE_KIND
        )
        assert select_root_solver_for_descriptor(case.descriptor) is NewtonRootSolver

    return _StepSelectionCase(
        name="implicit_stage_solve",
        descriptor=descriptor,
        order=2,
        state=state,
        rhs=rhs,
        exact=cases.exact_scalar_decay,
        tolerance=1.0e-7,
        ownership=SelectionOwnership(
            descriptor,
            time_integration_step_solve_regions(),
            solve_relation_parameter_schema(),
        ),
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
        exact=cases.exact_scalar_decay,
        tolerance=1.0e-8,
        ownership=SelectionOwnership(
            rhs_history_descriptor(),
            time_integration_step_map_regions(),
            map_structure_parameter_schema(),
        ),
        auto_selectable=False,
    )


def _nordsieck_history_selection_case() -> _StepSelectionCase:
    rhs = cases.scalar_decay_jacobian_rhs()
    integrator = _ti.MultistepIntegrator("adams", 4)
    state = integrator.init_state(rhs, 0.0, Tensor([1.0], backend=_TIME_BACKEND), 1e-2)
    descriptor = nordsieck_history_descriptor(1.0e-2)
    assert isinstance(state.history, _ti.NordsieckHistory)
    assert state.history.q == 4
    return _StepSelectionCase(
        name="nordsieck_history",
        descriptor=descriptor,
        order=4,
        state=state,
        rhs=rhs,
        exact=cases.exact_scalar_decay,
        tolerance=1.0e-8,
        ownership=SelectionOwnership(
            descriptor,
            time_integration_step_map_regions(),
            map_structure_parameter_schema(),
        ),
        auto_selectable=False,
        postcheck=lambda case, state, integrator: (
            assert_nordsieck_family(integrator, "adams")
        ),
    )


def assert_nordsieck_family(integrator: object, family: str) -> None:
    assert isinstance(integrator, _ti.MultistepIntegrator)
    assert integrator.family == family


def _assert_no_step_solve_relation(
    case: _StepSelectionCase,
    state: _ti.ODEState,
    integrator: object,
) -> None:
    candidate = getattr(integrator, "step_solve_relation_descriptor", None)
    if callable(candidate):
        with pytest.raises(ValueError):
            candidate(case.rhs, state, 1.0e-2)


def _composition_postcheck(
    case: _StepSelectionCase,
    state: _ti.ODEState,
    integrator: object,
) -> None:
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
    _assert_no_step_solve_relation(case, state, integrator)


def _uncertified_composition_postcheck(
    case: _StepSelectionCase, state: _ti.ODEState, integrator: object
) -> None:
    assert (
        case.descriptor.coordinate(MapStructureField.ADDITIVE_COMPONENT_COUNT).value
        == 2
    )
    assert not case.descriptor.coordinate(
        MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
    ).value
    assert math.isinf(
        case.descriptor.coordinate(
            MapStructureField.SYMPLECTIC_FORM_DEFECT_UPPER_BOUND
        ).value
    )
    _assert_no_step_solve_relation(case, state, integrator)


def _damped_composition_postcheck(
    case: _StepSelectionCase, state: _ti.ODEState, integrator: object
) -> None:
    assert (
        case.descriptor.coordinate(MapStructureField.ADDITIVE_COMPONENT_COUNT).value
        == 2
    )
    assert not case.descriptor.coordinate(
        MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
    ).value
    assert case.descriptor.coordinate(
        MapStructureField.SYMPLECTIC_FORM_DEFECT_UPPER_BOUND
    ).value == pytest.approx(math.sqrt(2.0) * _DAMPED_OSCILLATOR_GAMMA)
    _assert_no_step_solve_relation(case, state, integrator)


def _step_selection_cases() -> tuple[_StepSelectionCase, ...]:
    return (
        _map_selection_case(
            "rhs_evaluation",
            rhs_evaluation_descriptor(),
            4,
            _ti.BlackBoxRHS(lambda t, u: Tensor([-float(u[0])], backend=u.backend)),
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
            Tensor([1.0], backend=_TIME_BACKEND),
            cases.exact_scalar_decay,
            1.0e-7,
            lambda case, state, integrator: _assert_no_step_solve_relation(
                case, state, integrator
            ),
        ),
        _map_selection_case(
            "semilinear_map",
            semilinear_map_descriptor(),
            4,
            cases.semilinear_forcing_rhs(),
            Tensor([1.0], backend=_TIME_BACKEND),
            cases.exact_semilinear,
            1.0e-10,
            lambda case, state, integrator: _assert_no_step_solve_relation(
                case, state, integrator
            ),
        ),
        _map_selection_case(
            "hamiltonian_map",
            hamiltonian_map_descriptor(),
            4,
            cases.harmonic_hamiltonian_rhs(),
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
            cases.exact_ham,
            1.0e-10,
            lambda case, state, integrator: _assert_no_step_solve_relation(
                case, state, integrator
            ),
        ),
        _map_selection_case(
            "certified_composition_map",
            composite_map_descriptor_from_rhs(cases.oscillator_composite_rhs()),
            4,
            cases.oscillator_composite_rhs(),
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
            cases.exact_osc,
            1.0e-8,
            _composition_postcheck,
        ),
        _map_selection_case(
            "uncertified_composition_map",
            composite_map_descriptor_from_rhs(_uncertified_oscillator_composite_rhs()),
            4,
            _uncertified_oscillator_composite_rhs(),
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
            cases.exact_osc,
            1.0e-8,
            _uncertified_composition_postcheck,
        ),
        _map_selection_case(
            "damped_composition_map",
            composite_map_descriptor_from_rhs(_damped_oscillator_composite_rhs()),
            2,
            _damped_oscillator_composite_rhs(),
            Tensor([1.0, 0.0], backend=_TIME_BACKEND),
            _exact_damped_osc,
            2.0e-3,
            _damped_composition_postcheck,
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
        case.ownership.assert_owned_cell()
        request = AlgorithmRequest(
            requested_properties=frozenset({"one_step"}),
            order=case.order,
            descriptor=case.descriptor,
        )
        selected = _ti.select_time_integrator(request)
        assert selected.owner is case.owner
        integrator = selected.construct(request)
        assert type(integrator) is case.owner
        state = integrator.step(case.rhs, case.state, 1.0e-2)  # type: ignore[attr-defined]
        assert cases.err(state.u, case.exact, state.t) < case.tolerance
        case.postcheck(case, state, integrator)


class _StepSelectionRegionCoverageClaim(Claim[Any]):
    """Claim that selection calculations cover every step ownership region."""

    @property
    def description(self) -> str:
        return "correctness/step_selection_region_coverage"

    def check(self, _calibration: Any) -> None:
        cases_by_region = _step_selection_cases() + (_nordsieck_stiff_region_case(),)
        for region in (
            *time_integration_step_map_regions(),
            *time_integration_step_solve_regions(),
        ):
            assert any(
                region.owner is case.owner
                and region in case.ownership.regions
                and region.contains(case.descriptor)
                for case in cases_by_region
            ), region.owner.__name__


def _nordsieck_stiff_region_case() -> _StepSelectionCase:
    descriptor = nordsieck_history_descriptor(10.0)
    return _StepSelectionCase(
        name="nordsieck_stiff_history",
        descriptor=descriptor,
        order=4,
        state=_ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND)),
        rhs=cases.scalar_decay_jacobian_rhs(),
        exact=cases.exact_scalar_decay,
        tolerance=1.0,
        ownership=SelectionOwnership(
            descriptor,
            time_integration_step_map_regions(),
            map_structure_parameter_schema(),
        ),
        auto_selectable=False,
    )


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
        integrator = _ti.CompositionIntegrator(
            [_ti.RungeKuttaIntegrator(1), _ti.RungeKuttaIntegrator(1)],
            order=4,
        )
        certified = _ti.ODEState(0.0, Tensor([1.0, 0.0], backend=_TIME_BACKEND))
        uncertified = _ti.ODEState(0.0, Tensor([1.0, 0.0], backend=_TIME_BACKEND))
        dt = 5.0e-2
        steps = 200

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
        integrator = _ti.CompositionIntegrator(
            [_ti.RungeKuttaIntegrator(1), _ti.RungeKuttaIntegrator(1)],
            order=2,
        )
        state = _ti.ODEState(0.0, Tensor([1.0, 0.0], backend=_TIME_BACKEND))
        dt = 2.5e-3
        steps = 400

        initial_energy = _oscillator_energy(state.u)
        for _ in range(steps):
            state = integrator.step(rhs, state, dt)

        assert cases.err(state.u, _exact_damped_osc, state.t) < 2.0e-3
        assert _oscillator_energy(state.u) < initial_energy


class _StepDiagnosticStiffnessClaim(Claim[Any]):
    """Grounded claim for quantitative stiff/nonstiff map evidence."""

    @property
    def description(self) -> str:
        return "correctness/step_diagnostics_stiffness"

    def check(self, _calibration: Any) -> None:
        schema = map_structure_parameter_schema()
        regions = {region.name: region for region in schema.derived_regions}
        state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
        nonstiff = rhs_step_diagnostics_descriptor(
            _AffineDecayRHS(0.1),
            state,
            1.0e-1,
            local_error_target=1.0e-6,
            retry_budget=3,
        )
        stiff = rhs_step_diagnostics_descriptor(_AffineDecayRHS(100.0), state, 1.0e-1)
        for descriptor in (nonstiff, stiff):
            schema.validate_descriptor(descriptor)
            assert regions["single_step_rhs_evaluation"].contains(descriptor)
        assert regions["nonstiff_step"].contains(nonstiff)
        assert regions["stiff_step"].contains(stiff)
        assert nonstiff.coordinate(
            MapStructureField.STIFFNESS_ESTIMATE
        ).value == pytest.approx(1.0e-2)
        assert stiff.coordinate(
            MapStructureField.STIFFNESS_ESTIMATE
        ).value == pytest.approx(10.0)
        assert nonstiff.coordinate(
            MapStructureField.LOCAL_ERROR_TARGET
        ).value == pytest.approx(1.0e-6)
        assert nonstiff.coordinate(MapStructureField.RETRY_BUDGET).value == 3
        assert (
            nonstiff.coordinate(MapStructureField.RHS_EVALUATION_COST_FMAS).value == 1.0
        )
        domain_limited = rhs_step_diagnostics_descriptor(
            _BoundaryApproachRHS(),
            _ti.ODEState(0.0, Tensor([0.1], backend=_TIME_BACKEND)),
            1.0e-1,
        )
        schema.validate_descriptor(domain_limited)
        assert regions["domain_limited_step"].contains(domain_limited)
        assert domain_limited.coordinate(
            MapStructureField.DOMAIN_STEP_MARGIN
        ).value == pytest.approx(-0.1)


class _AutoIntegratorSelectionClaim(Claim[Any]):
    """Grounded claim for RHS-type projection into capability selection."""

    @property
    def description(self) -> str:
        return "correctness/auto_integrator_selects_capability_branches"

    def check(self, _calibration: Any) -> None:
        for case in _step_selection_cases():
            if not case.auto_selectable:
                continue
            auto = _ti.AutoIntegrator(case.order)
            assert auto.select(case.rhs).owner is case.owner
            state = auto.step(case.rhs, case.state, 1.0e-2)
            assert cases.err(state.u, case.exact, state.t) < case.tolerance

        with pytest.raises(ValueError, match="no algorithm"):
            _ti.AutoIntegrator(3).select(cases.harmonic_hamiltonian_rhs())


class _NordsieckFamilyFromStiffnessClaim(Claim[Any]):
    """Grounded claim that stiffness evidence selects the Nordsieck family."""

    @property
    def description(self) -> str:
        return "correctness/nordsieck_family_from_stiffness"

    def check(self, _calibration: Any) -> None:
        for stiffness, family in ((1.0e-2, "adams"), (10.0, "bdf")):
            request = AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=nordsieck_history_descriptor(stiffness),
            )
            selected = _ti.select_time_integrator(request)
            assert selected.owner is _ti.MultistepIntegrator
            assert_nordsieck_family(selected.construct(request), family)


class _DomainMarginAdvanceSelectionClaim(Claim[Any]):
    """Grounded claim that ordinary advance ownership does not split on margin."""

    @property
    def description(self) -> str:
        return "correctness/domain_margin_advance_selection"

    def check(self, _calibration: Any) -> None:
        interior_state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
        interior_step = rhs_step_diagnostics_descriptor(
            _AffineDecayRHS(0.1), interior_state, 1.0e-1
        )
        generic_interior = AlgorithmRequest(
            requested_properties=frozenset({"advance"}),
            order=3,
            descriptor=interior_step,
        )
        generic_interior_capability = _ti.select_time_integrator(generic_interior)
        assert generic_interior_capability.owner is _ti.IntegrationDriver

        boundary_state = _ti.ODEState(0.0, Tensor([0.1], backend=_TIME_BACKEND))
        boundary_step = rhs_step_diagnostics_descriptor(
            _BoundaryApproachRHS(), boundary_state, 1.0e-1
        )
        assert boundary_step.coordinate(
            MapStructureField.DOMAIN_STEP_MARGIN
        ).value == pytest.approx(-0.1)
        generic_limited = AlgorithmRequest(
            requested_properties=frozenset({"advance"}),
            order=3,
            descriptor=boundary_step,
        )
        generic_limited_capability = _ti.select_time_integrator(generic_limited)
        assert generic_limited_capability.owner is _ti.IntegrationDriver
        interior_region = self._selected_region(
            generic_interior_capability, interior_step
        )
        limited_region = self._selected_region(
            generic_limited_capability, boundary_step
        )
        assert interior_region == limited_region

        adaptive_limited_descriptor = ParameterDescriptor(
            derivative_oracle_descriptor().coordinates
            | {
                MapStructureField.DOMAIN_STEP_MARGIN: boundary_step.coordinate(
                    MapStructureField.DOMAIN_STEP_MARGIN
                )
            }
        )
        adaptive_limited = AlgorithmRequest(
            requested_properties=frozenset({"advance"}),
            order=2,
            descriptor=adaptive_limited_descriptor,
        )
        adaptive_limited_capability = _ti.select_time_integrator(adaptive_limited)
        assert adaptive_limited_capability.owner is _ti.AdaptiveNordsieckController
        assert self._selected_region(
            adaptive_limited_capability, adaptive_limited_descriptor
        ) == self._selected_region(
            adaptive_limited_capability,
            ParameterDescriptor(
                derivative_oracle_descriptor().coordinates
                | {
                    MapStructureField.DOMAIN_STEP_MARGIN: (
                        interior_step.coordinate(MapStructureField.DOMAIN_STEP_MARGIN)
                    )
                }
            ),
        )

    @staticmethod
    def _selected_region(capability: Any, descriptor: ParameterDescriptor) -> object:
        matches = tuple(
            region
            for region in capability.coverage_regions
            if region.contains(descriptor)
        )
        assert len(matches) == 1
        return matches[0]


class _AffineStageLinearSolveClaim(Claim[Any]):
    """Grounded claim for affine implicit-stage solve projection."""

    @property
    def description(self) -> str:
        return "correctness/affine_stage_linear_solve_projection"

    def check(self, _calibration: Any) -> None:
        integrator = _ti.ImplicitRungeKuttaIntegrator(2)
        state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
        dt = 1.0e-2
        step_descriptor = integrator.step_solve_relation_descriptor(
            _AffineDecayRHS(), state, dt
        )
        affine_relation = _ti.affine_stage_residual_relation(
            integrator, _AffineDecayRHS(), state, dt
        )
        assert isinstance(affine_relation, LinearResidualRelation)
        relation_descriptor = affine_relation.solve_relation_descriptor()
        for field in SolveRelationField:
            assert step_descriptor.coordinate(field) == relation_descriptor.coordinate(
                field
            )
        linear_descriptor = linear_operator_descriptor_from_solve_relation_descriptor(
            step_descriptor
        )
        assert select_linear_solver_for_descriptor(linear_descriptor) is DenseLUSolver
        evidence = assembled_linear_evidence_for(
            linear_descriptor,
            frozenset(LinearSolverField),
        )
        stage_value = DenseLUSolver().solve(evidence.operator, evidence.rhs)
        residual = evidence.operator.apply(stage_value) - evidence.rhs
        assert abs(float(residual[0])) < 1.0e-12
        expected_stage = 1.0 / (1.0 + 0.5 * dt)
        assert abs(float(stage_value[0]) - expected_stage) < 1.0e-12


class _ImexImplicitStageRootClaim(Claim[Any]):
    """Grounded claim for IMEX implicit-component root construction."""

    @property
    def description(self) -> str:
        return "correctness/imex_implicit_stage_root_relation"

    def check(self, _calibration: Any) -> None:
        rhs = cases.split_decay_rhs()
        y_exp = Tensor([1.0], backend=_TIME_BACKEND)
        gamma_dt = 0.25
        relation = _ti.imex_implicit_stage_root_relation(rhs, y_exp, 0.0, gamma_dt)
        assert isinstance(relation, FiniteDimensionalResidualRelation)
        descriptor = relation.solve_relation_descriptor()
        assert select_root_solver_for_descriptor(descriptor) is NewtonRootSolver
        stage_value = NewtonRootSolver().solve(relation)
        residual = relation.residual(stage_value)
        assert abs(float(residual[0])) < 1.0e-12
        assert abs(float(stage_value[0]) - (1.0 / (1.0 + 0.8 * gamma_dt))) < 1.0e-12

        state = _ti.ODEState(0.0, y_exp)
        stepped = _ti.AdditiveRungeKuttaIntegrator(2).step(rhs, state, 1.0e-2)
        assert cases.err(stepped.u, cases.exact_scalar_decay, stepped.t) < 1.0e-7


class _DirectionalDerivativeRootSolveClaim(Claim[Any]):
    """Grounded claim for nonlinear root solving with only JVP evidence."""

    @property
    def description(self) -> str:
        return "correctness/directional_derivative_root_solve"

    def check(self, _calibration: Any) -> None:
        def residual(x: Tensor) -> Tensor:
            return Tensor([float(x[0]) ** 2 - 2.0], backend=x.backend)

        def jvp(x: Tensor, v: Tensor) -> Tensor:
            return Tensor([2.0 * float(x[0]) * float(v[0])], backend=x.backend)

        relation = DirectionalDerivativeRootRelation(
            residual,
            jvp,
            Tensor([3.0], backend=_TIME_BACKEND),
        )
        direction = Tensor([0.25], backend=_TIME_BACKEND)
        assert abs(float(relation.jvp(relation.initial, direction)[0]) - 1.5) < 1e-12

        descriptor = relation.solve_relation_descriptor(map_linearity_defect=1.0)
        schema = solve_relation_parameter_schema()
        schema.validate_descriptor(descriptor)
        assert (
            descriptor.coordinate(SolveRelationField.DERIVATIVE_ORACLE_KIND).value
            == "jvp"
        )
        assert select_root_solver_for_descriptor(descriptor) is (
            MatrixFreeNewtonKrylovRootSolver
        )

        root = MatrixFreeNewtonKrylovRootSolver().solve(relation)
        assert abs(float(root[0]) ** 2 - 2.0) < 1e-10


class _JvpImplicitStageRootSolveClaim(Claim[Any]):
    """Grounded nonlinear implicit step with only JVP derivative evidence."""

    @property
    def description(self) -> str:
        return "correctness/jvp_implicit_stage_root_solve"

    def check(self, _calibration: Any) -> None:
        class QuadraticDecayJVP:
            def __call__(self, _t: float, u: Tensor) -> Tensor:
                return Tensor([-float(u[0]) ** 2], backend=u.backend)

            def jvp(self, _t: float, u: Tensor, v: Tensor) -> Tensor:
                return Tensor(
                    [-2.0 * float(u[0]) * float(v[0])],
                    backend=u.backend,
                )

        rhs = QuadraticDecayJVP()
        integrator = _ti.ImplicitRungeKuttaIntegrator(1)
        state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
        dt = 0.2

        descriptor = integrator.step_solve_relation_descriptor(rhs, state, dt)
        assert (
            descriptor.coordinate(SolveRelationField.DERIVATIVE_ORACLE_KIND).value
            == "jvp"
        )
        assert select_root_solver_for_descriptor(descriptor) is (
            MatrixFreeNewtonKrylovRootSolver
        )

        relation = _ti.implicit_stage_directional_derivative_root_relation(
            integrator,
            rhs,
            state,
            dt,
        )
        for field in SolveRelationField:
            assert descriptor.coordinate(field) == relation.solve_relation_descriptor(
                map_linearity_defect=None
            ).coordinate(field)
        root = MatrixFreeNewtonKrylovRootSolver().solve(relation)
        expected = (math.sqrt(1.0 + 4.0 * dt) - 1.0) / (2.0 * dt)
        assert abs(float(root[0]) - expected) < 1.0e-10
        assert abs(float(relation.residual(root)[0])) < 1.0e-10


_CORRECT_CLAIMS: tuple[Claim[Any], ...] = (
    _AutoIntegratorSelectionClaim(),
    _AffineStageLinearSolveClaim(),
    _ImexImplicitStageRootClaim(),
    _DirectionalDerivativeRootSolveClaim(),
    _JvpImplicitStageRootSolveClaim(),
    _StepSelectionRegionCoverageClaim(),
    *[_StepSelectionClaim(case) for case in _step_selection_cases()],
    _OscillatorInvariantComparisonClaim(),
    _OscillatorNegativeInvariantEvidenceClaim(),
    _DampedOscillatorSymplecticDefectClaim(),
    _StepDiagnosticStiffnessClaim(),
    _NordsieckFamilyFromStiffnessClaim(),
    _DomainMarginAdvanceSelectionClaim(),
)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[Any]) -> None:
    claim.check(None)
