"""Reaction-network calculation claims."""

from __future__ import annotations

from typing import Any

import pytest

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmRequest,
    LinearSolverField,
    MapStructureField,
    ParameterDescriptor,
    ReactionNetworkField,
    SolveRelationField,
    linear_solver_parameter_schema,
    map_structure_parameter_schema,
    reaction_network_parameter_schema,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.capabilities import (
    time_integration_step_map_regions,
)
from cosmic_foundry.theory.discrete import FiniteStateTransitionSystem
from tests.claims import Claim

_TIME_BACKEND = NumpyBackend()


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


def _finite_transition_initial_state(
    system: FiniteStateTransitionSystem,
) -> Tensor:
    return Tensor([1.0] + [0.0] * (system.state_count - 1), backend=_TIME_BACKEND)


def _finite_transition_conserved_form(
    system: FiniteStateTransitionSystem,
) -> Tensor:
    return Tensor(system.conserved_total_form(), backend=_TIME_BACKEND)


def _unit_transfer_rhs(
    system: FiniteStateTransitionSystem,
    rates: _ti.UnitTransferRates,
) -> _ti.ReactionNetworkRHS:
    return _ti.ReactionNetworkRHS.from_unit_transfer_system(
        system,
        rates,
        _finite_transition_initial_state(system),
    )


def _linear_form_annihilates_stoichiometry(
    linear_form: Tensor, stoichiometry: Tensor
) -> bool:
    product = linear_form @ stoichiometry
    return all(abs(float(product[j])) < 1e-12 for j in range(product.shape[0]))


class _ReactionChainIntegrationClaim(Claim[Any]):
    """Grounded claim for a finite reaction-chain projection."""

    @property
    def description(self) -> str:
        return "correctness/reaction_chain_projection_invariants"

    def check(self, _calibration: Any) -> None:
        system = FiniteStateTransitionSystem.chain(3)
        u0 = _finite_transition_initial_state(system)
        rhs = _unit_transfer_rhs(system, (80.0, 12.0))
        state = _ti.ODEState(0.0, u0)
        integrator = _ti.ImplicitRungeKuttaIntegrator(2)
        auto = _ti.AutoIntegrator(2)

        assert auto.select(rhs).implementation == "ImplicitRungeKuttaIntegrator"

        reaction_descriptor = rhs.reaction_network_descriptor()
        map_descriptor = rhs.map_structure_descriptor()
        reaction_schema = reaction_network_parameter_schema()
        map_schema = map_structure_parameter_schema()
        reaction_schema.validate_descriptor(reaction_descriptor)
        map_schema.validate_descriptor(map_descriptor)
        reaction_regions = {
            region.name: region for region in reaction_schema.derived_regions
        }
        assert reaction_regions["conserved_network"].contains(reaction_descriptor)
        assert (
            reaction_descriptor.coordinate(ReactionNetworkField.SPECIES_COUNT).value
            == 3
        )
        assert (
            reaction_descriptor.coordinate(ReactionNetworkField.REACTION_COUNT).value
            == 2
        )
        assert (
            reaction_descriptor.coordinate(
                ReactionNetworkField.STOICHIOMETRY_RANK
            ).value
            == 2
        )
        assert (
            reaction_descriptor.coordinate(
                ReactionNetworkField.CONSERVATION_LAW_COUNT
            ).value
            == 1
        )
        assert (
            map_descriptor.coordinate(
                MapStructureField.CONSERVED_LINEAR_FORM_COUNT
            ).value
            == reaction_descriptor.coordinate(
                ReactionNetworkField.CONSERVATION_LAW_COUNT
            ).value
        )
        assert (
            _ti.select_time_integrator(
                AlgorithmRequest(
                    requested_properties=frozenset({"one_step"}),
                    order=4,
                    descriptor=map_descriptor,
                )
            ).implementation
            == "RungeKuttaIntegrator"
        )
        _assert_owned_step_map_cell(map_descriptor, _ti.RungeKuttaIntegrator)
        assert (
            reaction_descriptor.coordinate(
                ReactionNetworkField.EQUILIBRIUM_CONSTRAINT_COUNT
            ).value
            == 2
        )
        constraint_descriptor = rhs.constraint_aware_descriptor()
        constraint_owner = _ti.select_time_integrator(
            AlgorithmRequest(
                requested_properties=frozenset({"advance"}),
                order=2,
                descriptor=constraint_descriptor,
            )
        )
        assert constraint_owner.implementation == "ConstraintAwareController"

        descriptor = integrator.step_solve_relation_descriptor(rhs, state, 2.0e-3)
        schema = solve_relation_parameter_schema()
        schema.validate_descriptor(descriptor)
        regions = {region.name: region for region in schema.derived_regions}

        assert regions["linear_system"].contains(descriptor)
        assert descriptor.coordinate(SolveRelationField.DIM_X).value == (
            state.u.shape[0] * len(integrator.A_sym)
        )
        assert (
            descriptor.coordinate(SolveRelationField.MAP_LINEARITY_DEFECT).value == 0.0
        )
        linear_descriptor = integrator.step_linear_operator_descriptor(
            rhs, state, 2.0e-3
        )
        linear_schema = linear_solver_parameter_schema()
        linear_schema.validate_descriptor(linear_descriptor.parameter_descriptor)
        linear_regions = {
            region.name: region for region in linear_schema.derived_regions
        }
        assert linear_regions["linear_system"].contains(
            linear_descriptor.parameter_descriptor
        )
        assert linear_regions["full_rank"].contains(
            linear_descriptor.parameter_descriptor
        )
        assert linear_descriptor.coordinate(LinearSolverField.RANK_ESTIMATE).value == (
            state.u.shape[0] * len(integrator.A_sym)
        )

        invariant = _finite_transition_conserved_form(system)
        assert _linear_form_annihilates_stoichiometry(
            invariant, rhs.stoichiometry_matrix
        )
        initial_invariant = float(invariant @ state.u)

        for _ in range(20):
            state = integrator.step(rhs, state, 2.0e-3)
            assert rhs.state_domain.check(state.u).accepted

        final_invariant = float(invariant @ state.u)
        assert abs(final_invariant - initial_invariant) < 1e-10
        assert (
            abs(sum(float(state.u[i]) for i in range(state.u.shape[0])) - 1.0) < 1e-10
        )

        controller = _ti.ConstraintAwareController(
            rhs=rhs,
            integrator=integrator,
            inner=_ti.PIController(
                alpha=0.35,
                beta=0.2,
                tol=1e-7,
                dt0=2.0e-3,
                factor_max=1.15,
            ),
        )
        controlled_state = controller.advance(
            u0,
            0.0,
            2.0e-2,
        )
        assert rhs.state_domain.check(controlled_state.u).accepted
        assert abs(float(invariant @ controlled_state.u) - initial_invariant) < 1e-10


class _BranchedFiniteTransitionNetworkClaim(Claim[Any]):
    """Grounded claim for a branched finite transition-system projection."""

    @property
    def description(self) -> str:
        return "correctness/branched_finite_transition_projection_invariants"

    def check(self, _calibration: Any) -> None:
        transition_system = FiniteStateTransitionSystem(
            4,
            ((0, 1), (0, 2), (2, 3)),
        )
        rhs = _unit_transfer_rhs(transition_system, (25.0, 5.0, 7.0))
        state = _ti.ODEState(0.0, _finite_transition_initial_state(transition_system))
        controller = _ti.IntegrationDriver(
            _ti.RungeKuttaIntegrator(4),
            controller=_ti.PIController(
                alpha=0.35,
                beta=0.2,
                tol=1e-7,
                dt0=1e-3,
            ),
        )

        reaction_descriptor = rhs.reaction_network_descriptor()
        reaction_network_parameter_schema().validate_descriptor(reaction_descriptor)
        assert reaction_descriptor.coordinate(
            ReactionNetworkField.SPECIES_COUNT
        ).value == (transition_system.state_count)
        assert reaction_descriptor.coordinate(
            ReactionNetworkField.REACTION_COUNT
        ).value == (transition_system.transition_count)
        assert transition_system.stoichiometry_matrix() == tuple(
            tuple(
                int(float(rhs.stoichiometry_matrix[i, j]))
                for j in range(transition_system.transition_count)
            )
            for i in range(transition_system.state_count)
        )
        invariant = _finite_transition_conserved_form(transition_system)
        assert _linear_form_annihilates_stoichiometry(
            invariant,
            rhs.stoichiometry_matrix,
        )
        initial_invariant = float(invariant @ state.u)

        state = controller.advance(rhs, state.u, state.t, 0.08)

        assert rhs.state_domain.check(state.u).accepted
        assert abs(float(invariant @ state.u) - initial_invariant) < 1e-9
        assert float(state.u[1]) > float(state.u[2]) > float(state.u[3]) > 0.0


_CORRECT_CLAIMS: tuple[Claim[Any], ...] = (
    _ReactionChainIntegrationClaim(),
    _BranchedFiniteTransitionNetworkClaim(),
)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[Any]) -> None:
    claim.check(None)
