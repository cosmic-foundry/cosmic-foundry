"""Reaction-network calculation claims."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmRequest,
    AssembledLinearEvidence,
    LinearSolverField,
    MapStructureField,
    ParameterDescriptor,
    ReactionNetworkField,
    SolveRelationField,
    assembled_linear_evidence_for,
    linear_operator_descriptor_from_solve_relation_descriptor,
    linear_solver_parameter_schema,
    map_structure_parameter_schema,
    reaction_network_parameter_schema,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.solvers import select_linear_solver_for_descriptor
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.capabilities import (
    derivative_oracle_descriptor,
    time_integration_capabilities,
    time_integration_step_map_regions,
)
from cosmic_foundry.theory.discrete import FiniteStateTransitionSystem
from tests.claims import Claim
from tests.selection_ownership import SelectionOwnership

_TIME_BACKEND = NumpyBackend()


def _time_integration_ownership(
    descriptor: ParameterDescriptor,
    *,
    requested_properties: frozenset[str],
    order: int,
) -> SelectionOwnership:
    return SelectionOwnership.from_request(
        AlgorithmRequest(
            requested_properties=requested_properties,
            order=order,
            descriptor=descriptor,
        ),
        time_integration_capabilities(),
    )


def _step_map_ownership(descriptor: ParameterDescriptor) -> SelectionOwnership:
    return SelectionOwnership(
        descriptor,
        time_integration_step_map_regions(),
        map_structure_parameter_schema(),
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


def _residual_norm(residual: Tensor) -> float:
    return math.sqrt(sum(float(residual[i]) ** 2 for i in range(residual.shape[0])))


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

        step_descriptor = integrator.step_solve_relation_descriptor(rhs, state, 2.0e-3)
        assert (
            auto.select(rhs).owner
            is _time_integration_ownership(
                derivative_oracle_descriptor(),
                requested_properties=frozenset({"one_step"}),
                order=2,
            ).owner
        )

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
            ).owner
            is _step_map_ownership(map_descriptor).owner
        )
        _step_map_ownership(map_descriptor).assert_owned_cell()
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
        assert (
            constraint_owner.owner
            is _time_integration_ownership(
                constraint_descriptor,
                requested_properties=frozenset({"advance"}),
                order=2,
            ).owner
        )

        descriptor = step_descriptor
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
        linear_descriptor = linear_operator_descriptor_from_solve_relation_descriptor(
            descriptor
        )
        linear_schema = linear_solver_parameter_schema()
        linear_schema.validate_descriptor(linear_descriptor)
        linear_regions = {
            region.name: region for region in linear_schema.derived_regions
        }
        assert linear_regions["linear_system"].contains(linear_descriptor)
        assert linear_regions["full_rank"].contains(linear_descriptor)
        assert linear_descriptor.coordinate(LinearSolverField.RANK_ESTIMATE).value == (
            state.u.shape[0] * len(integrator.A_sym)
        )
        assert descriptor.coordinate(SolveRelationField.ACCEPTANCE_RELATION).value == (
            linear_descriptor.coordinate(SolveRelationField.ACCEPTANCE_RELATION).value
        )
        assert (
            descriptor.coordinate(SolveRelationField.REQUESTED_RESIDUAL_TOLERANCE).value
            == linear_descriptor.coordinate(
                SolveRelationField.REQUESTED_RESIDUAL_TOLERANCE
            ).value
        )
        selected_solver = select_linear_solver_for_descriptor(linear_descriptor)
        linear_evidence = assembled_linear_evidence_for(
            linear_descriptor, frozenset(LinearSolverField)
        )
        assert isinstance(linear_evidence, AssembledLinearEvidence)
        stage_solution = selected_solver().solve(
            linear_evidence.operator,
            linear_evidence.rhs,
        )
        residual = linear_evidence.operator.apply(stage_solution) - linear_evidence.rhs
        assert (
            _residual_norm(residual)
            <= linear_descriptor.coordinate(
                SolveRelationField.REQUESTED_RESIDUAL_TOLERANCE
            ).value
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


class _TransientReactionEquilibriumClaim(Claim[Any]):
    """Grounded claim for transient approach to a reaction-network equilibrium."""

    @property
    def description(self) -> str:
        return "correctness/nse/transient"

    def check(self, _calibration: Any) -> None:
        rate = 5.0
        rhs = _ti.ReactionNetworkRHS(
            Tensor([[-1.0], [1.0]], backend=_TIME_BACKEND),
            lambda t, u: Tensor([rate * float(u[0])], backend=u.backend),
            lambda t, u: Tensor([rate * float(u[1])], backend=u.backend),
            Tensor([0.9, 0.1], backend=_TIME_BACKEND),
        )
        controller = _ti.ConstraintAwareController(
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

        state = controller.advance(
            Tensor([0.7, 0.3], backend=_TIME_BACKEND),
            0.0,
            0.1,
        )
        expected_0 = 0.5 + 0.2 * math.exp(-2.0 * rate * state.t)

        assert abs(float(state.u[0]) - expected_0) < 1e-4
        assert abs(float(state.u[1]) - (1.0 - expected_0)) < 1e-4
        assert abs(float(state.u[0]) + float(state.u[1]) - 1.0) < 1e-10
        assert rhs.state_domain.check(state.u).accepted


@dataclass
class _EquilibriumNetworkSpec:
    name: str
    topo: str
    n: int
    rates: list[float]
    expected_walltime_s: float = 1.0

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
        stoichiometry = Tensor(rows, backend=_TIME_BACKEND)

        def forward_rates(t: float, u: Tensor) -> Tensor:
            idx = lambda j: j if topo == "chain" else 0  # noqa: E731
            return Tensor(
                [rates[j] * float(u[idx(j)]) for j in range(p)],
                backend=u.backend,
            )

        def reverse_rates(t: float, u: Tensor) -> Tensor:
            return Tensor(
                [rates[j] * float(u[j + 1]) for j in range(p)],
                backend=u.backend,
            )

        return _ti.ReactionNetworkRHS(
            stoichiometry,
            forward_rates,
            reverse_rates,
            self.u0(),
        )


def _chain_specs(
    nr: range,
    k: float = 1.0,
    *,
    expected_walltime_s: float = 1.0,
) -> list[_EquilibriumNetworkSpec]:
    return [
        _EquilibriumNetworkSpec(
            f"chain-n{n}-k{k:.0f}",
            "chain",
            n,
            [k] * (n - 1),
            expected_walltime_s,
        )
        for n in nr
    ]


def _spoke_specs(
    nr: range,
    ks: list[int],
    *,
    expected_walltime_s: float = 1.0,
) -> list[_EquilibriumNetworkSpec]:
    specs = []
    for n in nr:
        p, fast_reactions = n - 1, (n - 1) // 2
        for k in ks:
            specs.append(
                _EquilibriumNetworkSpec(
                    f"spoke-n{n}-k{k:.0f}",
                    "spoke",
                    n,
                    [float(k)] * fast_reactions + [1.0] * (p - fast_reactions),
                    expected_walltime_s,
                )
            )
    return specs


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


class _EquilibriumNetworkTargetClaim(Claim[Any]):
    """Grounded claim for convergence to a network statistical equilibrium."""

    def __init__(self, spec: _EquilibriumNetworkSpec) -> None:
        self._spec = spec

    @property
    def description(self) -> str:
        return f"correctness/nse/{self._spec.name}"

    @property
    def expected_walltime_s(self) -> float:
        return self._spec.expected_walltime_s

    def check(self, _calibration: Any) -> None:
        self.skip_if_over_walltime_budget()
        rhs = self._spec.build_rhs()
        if self._spec.topo == "chain":
            controller = _adaptive_nordsieck_controller()
            state = controller.advance(
                rhs,
                self._spec.u0(),
                0.0,
                self._spec.t_end(),
                self._spec.dt0(),
            )
        else:
            controller = _ti.ConstraintAwareController(
                rhs=rhs,
                integrator=_ti.ImplicitRungeKuttaIntegrator(2),
                inner=_ti.PIController(
                    alpha=0.35,
                    beta=0.2,
                    tol=1e-5,
                    dt0=self._spec.dt0(),
                    factor_max=1.15,
                ),
                eps_activate=0.01,
                eps_deactivate=0.1,
            )
            state = controller.advance(
                self._spec.u0(),
                0.0,
                self._spec.t_end(),
                stop_at_nse=True,
            )

        expected = 1.0 / self._spec.n
        for i in range(self._spec.n):
            assert abs(float(state.u[i]) - expected) < 1e-6
        assert abs(sum(float(state.u[i]) for i in range(self._spec.n)) - 1.0) < 1e-10


_CI_EQUILIBRIUM_SPECS = _chain_specs(range(3, 5)) + _spoke_specs(range(3, 7), [1, 10])
_OFF_EQUILIBRIUM_SPECS = _chain_specs(
    range(5, 12), expected_walltime_s=5.0
) + _spoke_specs(range(7, 22), [1, 10, 100], expected_walltime_s=5.0)


_CORRECT_CLAIMS: tuple[Claim[Any], ...] = (
    _ReactionChainIntegrationClaim(),
    _BranchedFiniteTransitionNetworkClaim(),
    _TransientReactionEquilibriumClaim(),
    *[_EquilibriumNetworkTargetClaim(spec) for spec in _CI_EQUILIBRIUM_SPECS],
    *[_EquilibriumNetworkTargetClaim(spec) for spec in _OFF_EQUILIBRIUM_SPECS],
)


@pytest.mark.parametrize(
    "claim", _CORRECT_CLAIMS, ids=[c.description for c in _CORRECT_CLAIMS]
)
def test_correctness(claim: Claim[Any]) -> None:
    claim.check(None)
