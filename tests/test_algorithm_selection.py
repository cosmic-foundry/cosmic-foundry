"""Algorithm-selection calculation claims."""

from __future__ import annotations

import math
from typing import Any

import pytest

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmRequest,
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


def _assert_owned_step_map_cell(
    descriptor: ParameterDescriptor,
    owner: type,
) -> None:
    schema = map_structure_parameter_schema()
    regions = time_integration_step_map_regions()

    schema.validate_descriptor(descriptor)
    assert schema.cell_status(descriptor, regions) == "owned"
    assert tuple(region.owner for region in regions if region.contains(descriptor)) == (
        owner,
    )


def _assert_owned_step_solve_cell(
    descriptor: ParameterDescriptor,
    owner: type,
) -> None:
    schema = solve_relation_parameter_schema()
    regions = time_integration_step_solve_regions()

    schema.validate_descriptor(descriptor)
    assert schema.cell_status(descriptor, regions) == "owned"
    assert tuple(region.owner for region in regions if region.contains(descriptor)) == (
        owner,
    )


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


class _HistoryStateSelectionClaim(Claim[Any]):
    """Grounded claim for history-dependent method ownership."""

    @property
    def description(self) -> str:
        return "correctness/history_state_selection"

    def check(self, _calibration: Any) -> None:
        rhs = cases.scalar_decay_jacobian_rhs()
        u0 = Tensor([1.0], backend=_TIME_BACKEND)
        dt = 1.0e-2

        ab = _ti.ExplicitMultistepIntegrator.for_order(4)
        state = _ti.ODEState(0.0, u0)
        for _ in range(3):
            state = ab.step(rhs, state, dt)
        assert isinstance(state.history, tuple)
        assert len(state.history) == 3
        selected_ab = _ti.select_time_integrator(
            AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=rhs_history_descriptor(),
            )
        )
        assert selected_ab.implementation == _ti.ExplicitMultistepIntegrator.__name__
        _assert_owned_step_map_cell(
            rhs_history_descriptor(),
            _ti.ExplicitMultistepIntegrator,
        )
        state = ab.step(rhs, state, dt)
        assert cases.err(state.u, cases.exact_scalar_decay, state.t) < 1.0e-8

        nordsieck = _ti.MultistepIntegrator("adams", 4)
        nordsieck_state = nordsieck.init_state(rhs, 0.0, u0, dt)
        assert isinstance(nordsieck_state.history, _ti.NordsieckHistory)
        assert nordsieck_state.history.q == 4
        selected_nordsieck = _ti.select_time_integrator(
            AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=nordsieck_history_descriptor(),
            )
        )
        assert selected_nordsieck.implementation == _ti.MultistepIntegrator.__name__
        _assert_owned_step_map_cell(
            nordsieck_history_descriptor(),
            _ti.MultistepIntegrator,
        )
        nordsieck_state = nordsieck.step(rhs, nordsieck_state, dt)
        assert (
            cases.err(nordsieck_state.u, cases.exact_scalar_decay, nordsieck_state.t)
            < 1.0e-8
        )


class _ImplicitStageSolveSelectionClaim(Claim[Any]):
    """Grounded claim for derivative-oracle ownership as a step solve."""

    @property
    def description(self) -> str:
        return "correctness/implicit_stage_solve_selection"

    def check(self, _calibration: Any) -> None:
        rhs = cases.scalar_decay_jacobian_rhs()
        integrator = _ti.ImplicitRungeKuttaIntegrator(2)
        state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
        dt = 1.0e-2

        selected = _ti.select_time_integrator(
            AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=2,
                descriptor=derivative_oracle_descriptor(),
            )
        )
        assert selected.implementation == _ti.ImplicitRungeKuttaIntegrator.__name__

        descriptor = integrator.step_solve_relation_descriptor(rhs, state, dt)
        _assert_owned_step_solve_cell(descriptor, _ti.ImplicitRungeKuttaIntegrator)
        assert descriptor.coordinate(
            SolveRelationField.DERIVATIVE_ORACLE_KIND
        ) == derivative_oracle_descriptor().coordinate(
            SolveRelationField.DERIVATIVE_ORACLE_KIND
        )

        state = integrator.step(rhs, state, dt)
        assert cases.err(state.u, cases.exact_scalar_decay, state.t) < 1.0e-7


class _SplitStepMapSelectionClaim(Claim[Any]):
    """Grounded claim for split-step ownership as map composition evidence."""

    @property
    def description(self) -> str:
        return "correctness/split_step_map_selection"

    def check(self, _calibration: Any) -> None:
        rhs = cases.split_decay_rhs()
        integrator = _ti.AdditiveRungeKuttaIntegrator(2)
        state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
        dt = 1.0e-2

        selected = _ti.select_time_integrator(
            AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=2,
                descriptor=split_map_descriptor(),
            )
        )
        assert selected.implementation == _ti.AdditiveRungeKuttaIntegrator.__name__
        _assert_owned_step_map_cell(
            split_map_descriptor(),
            _ti.AdditiveRungeKuttaIntegrator,
        )
        with pytest.raises(ValueError):
            integrator.step_solve_relation_descriptor(rhs, state, dt)

        state = integrator.step(rhs, state, dt)
        assert cases.err(state.u, cases.exact_scalar_decay, state.t) < 1.0e-7


class _SemilinearStepMapSelectionClaim(Claim[Any]):
    """Grounded claim for semilinear ownership as map composition evidence."""

    @property
    def description(self) -> str:
        return "correctness/semilinear_step_map_selection"

    def check(self, _calibration: Any) -> None:
        rhs = cases.semilinear_forcing_rhs()
        integrator = _ti.LawsonRungeKuttaIntegrator(4)
        state = _ti.ODEState(0.0, Tensor([1.0], backend=_TIME_BACKEND))
        dt = 1.0e-2

        selected = _ti.select_time_integrator(
            AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=semilinear_map_descriptor(),
            )
        )
        assert selected.implementation == _ti.LawsonRungeKuttaIntegrator.__name__
        _assert_owned_step_map_cell(
            semilinear_map_descriptor(),
            _ti.LawsonRungeKuttaIntegrator,
        )
        assert not callable(getattr(integrator, "step_solve_relation_descriptor", None))
        assert not callable(
            getattr(integrator, "step_linear_operator_descriptor", None)
        )

        state = integrator.step(rhs, state, dt)
        assert cases.err(state.u, cases.exact_semilinear, state.t) < 1.0e-10


class _HamiltonianStepMapSelectionClaim(Claim[Any]):
    """Grounded claim for Hamiltonian ownership as map partition evidence."""

    @property
    def description(self) -> str:
        return "correctness/hamiltonian_step_map_selection"

    def check(self, _calibration: Any) -> None:
        rhs = cases.harmonic_hamiltonian_rhs()
        integrator = _ti.SymplecticCompositionIntegrator(4)
        state = _ti.ODEState(0.0, Tensor([1.0, 0.0], backend=_TIME_BACKEND))
        dt = 1.0e-2

        selected = _ti.select_time_integrator(
            AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=hamiltonian_map_descriptor(),
            )
        )
        assert selected.implementation == _ti.SymplecticCompositionIntegrator.__name__
        _assert_owned_step_map_cell(
            hamiltonian_map_descriptor(),
            _ti.SymplecticCompositionIntegrator,
        )
        with pytest.raises(ValueError):
            integrator.step_solve_relation_descriptor(rhs, state, dt)
        with pytest.raises(ValueError):
            integrator.step_linear_operator_descriptor(rhs, state, dt)

        state = integrator.step(rhs, state, dt)
        assert cases.err(state.u, cases.exact_ham, state.t) < 1.0e-10


class _CompositionStepMapSelectionClaim(Claim[Any]):
    """Grounded claim for composition ownership as component-flow evidence."""

    @property
    def description(self) -> str:
        return "correctness/composition_step_map_selection"

    def check(self, _calibration: Any) -> None:
        rhs = cases.oscillator_composite_rhs()
        integrator = _ti.CompositionIntegrator(
            [_ti.RungeKuttaIntegrator(1), _ti.RungeKuttaIntegrator(1)],
            order=4,
        )
        state = _ti.ODEState(0.0, Tensor([1.0, 0.0], backend=_TIME_BACKEND))
        dt = 1.0e-2
        descriptor = composite_map_descriptor_from_rhs(rhs)

        selected = _ti.select_time_integrator(
            AlgorithmRequest(
                requested_properties=frozenset({"one_step"}),
                order=4,
                descriptor=descriptor,
            )
        )
        assert selected.implementation == _ti.CompositionIntegrator.__name__
        _assert_owned_step_map_cell(descriptor, _ti.CompositionIntegrator)
        assert (
            descriptor.coordinate(MapStructureField.ADDITIVE_COMPONENT_COUNT).value == 2
        )
        assert not descriptor.coordinate(
            MapStructureField.HAMILTONIAN_PARTITION_AVAILABLE
        ).value
        assert descriptor.coordinate(
            MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
        ).value
        generic_descriptor = composite_map_descriptor(len(rhs.components))
        assert not generic_descriptor.coordinate(
            MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE
        ).value
        with pytest.raises(ValueError):
            integrator.step_solve_relation_descriptor(rhs, state, dt)
        with pytest.raises(ValueError):
            integrator.step_linear_operator_descriptor(rhs, state, dt)

        state = integrator.step(rhs, state, dt)
        assert cases.err(state.u, cases.exact_osc, state.t) < 1.0e-8


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
            assert (
                _ti.select_time_integrator(
                    AlgorithmRequest(
                        requested_properties=frozenset({"one_step"}),
                        order=4,
                        descriptor=descriptor,
                    )
                ).implementation
                == _ti.CompositionIntegrator.__name__
            )
            _assert_owned_step_map_cell(descriptor, _ti.CompositionIntegrator)

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
            == _ti.CompositionIntegrator.__name__
        )
        _assert_owned_step_map_cell(descriptor, _ti.CompositionIntegrator)

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
    _HistoryStateSelectionClaim(),
    _ImplicitStageSolveSelectionClaim(),
    _SplitStepMapSelectionClaim(),
    _SemilinearStepMapSelectionClaim(),
    _HamiltonianStepMapSelectionClaim(),
    _CompositionStepMapSelectionClaim(),
    _OscillatorInvariantComparisonClaim(),
    _OscillatorNegativeInvariantEvidenceClaim(),
    _DampedOscillatorSymplecticDefectClaim(),
)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[Any]) -> None:
    claim.check(None)
