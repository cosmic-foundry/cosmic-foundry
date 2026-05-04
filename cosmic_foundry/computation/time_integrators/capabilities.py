"""Algorithm structure contracts for time-integration selection."""

from __future__ import annotations

from typing import Literal

from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmCapability,
    AlgorithmRegistry,
    AlgorithmRequest,
    AlgorithmStructureContract,
    ComparisonPredicate,
    CoverageRegion,
    DescriptorCoordinate,
    DescriptorField,
    EvidenceSource,
    MapStructureField,
    MembershipPredicate,
    ParameterDescriptor,
    SolveRelationField,
)
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.adaptive_nordsieck import (
    AdaptiveNordsieckController,
)
from cosmic_foundry.computation.time_integrators.constraint_aware import (
    ConstraintAwareController,
)
from cosmic_foundry.computation.time_integrators.domains import (
    predict_domain_step_limit,
)
from cosmic_foundry.computation.time_integrators.explicit_multistep import (
    ExplicitMultistepIntegrator,
)
from cosmic_foundry.computation.time_integrators.exponential import (
    LawsonRungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.imex import (
    AdditiveRungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.implicit import (
    ImplicitRungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.integration_driver import (
    IntegrationDriver,
)
from cosmic_foundry.computation.time_integrators.integrator import ODEState
from cosmic_foundry.computation.time_integrators.nordsieck import (
    MultistepIntegrator,
)
from cosmic_foundry.computation.time_integrators.runge_kutta import (
    RungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.splitting import (
    ComponentFlowProtocol,
    CompositeRHSProtocol,
    CompositionIntegrator,
)
from cosmic_foundry.computation.time_integrators.stiffness import StiffnessDiagnostic
from cosmic_foundry.computation.time_integrators.symplectic import (
    SymplecticCompositionIntegrator,
)

TimeIntegrationCapability = AlgorithmCapability
TimeIntegrationRegistry = AlgorithmRegistry


def _contract(
    *,
    provides: tuple[str, ...],
) -> AlgorithmStructureContract:
    return AlgorithmStructureContract(frozenset(), frozenset(provides))


def _map_structure_coordinates(
    overrides: dict[DescriptorField, DescriptorCoordinate] | None = None,
) -> dict[DescriptorField, DescriptorCoordinate]:
    field = MapStructureField
    coordinates: dict[DescriptorField, DescriptorCoordinate] = {
        field.RHS_EVALUATION_AVAILABLE: DescriptorCoordinate(False),
        field.RHS_HISTORY_AVAILABLE: DescriptorCoordinate(False),
        field.NORDSIECK_HISTORY_AVAILABLE: DescriptorCoordinate(False),
        field.EXACT_LINEAR_OPERATOR_AVAILABLE: DescriptorCoordinate(False),
        field.NONLINEAR_RESIDUAL_AVAILABLE: DescriptorCoordinate(False),
        field.EXPLICIT_COMPONENT_AVAILABLE: DescriptorCoordinate(False),
        field.IMPLICIT_COMPONENT_AVAILABLE: DescriptorCoordinate(False),
        field.IMPLICIT_COMPONENT_DERIVATIVE_ORACLE_KIND: DescriptorCoordinate(
            "unavailable"
        ),
        field.HAMILTONIAN_PARTITION_AVAILABLE: DescriptorCoordinate(False),
        field.SYMPLECTIC_FORM_DEFECT_UPPER_BOUND: DescriptorCoordinate(float("inf")),
        field.SYMPLECTIC_FORM_INVARIANT_AVAILABLE: DescriptorCoordinate(False),
        field.CONSERVED_LINEAR_FORM_COUNT: DescriptorCoordinate(0),
        field.ALGEBRAIC_CONSTRAINT_COUNT: DescriptorCoordinate(0),
        field.ADDITIVE_COMPONENT_COUNT: DescriptorCoordinate(0),
        field.DOMAIN_STEP_MARGIN: DescriptorCoordinate(float("inf")),
        field.STIFFNESS_ESTIMATE: DescriptorCoordinate(0.0),
        field.LOCAL_ERROR_TARGET: DescriptorCoordinate(1.0e-8),
        field.RETRY_BUDGET: DescriptorCoordinate(0),
        field.RHS_EVALUATION_COST_FMAS: DescriptorCoordinate(0.0),
    }
    if overrides is not None:
        coordinates.update(overrides)
    return coordinates


def derivative_oracle_descriptor() -> ParameterDescriptor:
    """Return solve-relation evidence for an available Jacobian callback."""
    return ParameterDescriptor(
        {
            SolveRelationField.DERIVATIVE_ORACLE_KIND: DescriptorCoordinate(
                "jacobian_callback"
            )
        }
    )


def rhs_evaluation_descriptor() -> ParameterDescriptor:
    """Return map evidence for direct RHS evaluation."""
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {field.RHS_EVALUATION_AVAILABLE: DescriptorCoordinate(True)}
        )
    )


def rhs_step_diagnostics_descriptor(
    rhs: object,
    state: ODEState,
    dt: float,
    *,
    local_error_target: float = 1.0e-8,
    retry_budget: int = 0,
) -> ParameterDescriptor:
    """Return quantitative map evidence for one proposed RHS step."""
    if dt <= 0.0:
        raise ValueError("step diagnostics require dt > 0")
    if local_error_target <= 0.0:
        raise ValueError("local error target must be positive")
    if retry_budget < 0:
        raise ValueError("retry budget must be nonnegative")
    t = float(state.t)
    u = state.u
    if not isinstance(u, Tensor):
        raise TypeError("step diagnostics require a Tensor state vector")
    jacobian = getattr(rhs, "jacobian", None)
    stiffness = 0.0
    stiffness_evidence: EvidenceSource = "unavailable"
    if callable(jacobian):
        stiffness = StiffnessDiagnostic().update(jacobian(t, u), dt)
        stiffness_evidence = "upper_bound"
    step_limit = predict_domain_step_limit(rhs, t, u)
    domain_step_margin = (
        float("inf") if step_limit is None else (float(step_limit) / dt) - 1.0
    )
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.RHS_EVALUATION_AVAILABLE: DescriptorCoordinate(True),
                field.STIFFNESS_ESTIMATE: DescriptorCoordinate(
                    stiffness, evidence=stiffness_evidence
                ),
                field.DOMAIN_STEP_MARGIN: DescriptorCoordinate(
                    domain_step_margin, evidence="estimate"
                ),
                field.LOCAL_ERROR_TARGET: DescriptorCoordinate(local_error_target),
                field.RETRY_BUDGET: DescriptorCoordinate(retry_budget),
                field.RHS_EVALUATION_COST_FMAS: DescriptorCoordinate(
                    float(max(1, u.shape[0]))
                ),
            }
        )
    )


def conserved_rhs_evaluation_descriptor(
    conserved_linear_form_count: int,
) -> ParameterDescriptor:
    """Return map evidence for direct RHS evaluation with linear invariants."""
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.RHS_EVALUATION_AVAILABLE: DescriptorCoordinate(True),
                field.CONSERVED_LINEAR_FORM_COUNT: DescriptorCoordinate(
                    conserved_linear_form_count
                ),
            }
        )
    )


def rhs_history_descriptor() -> ParameterDescriptor:
    """Return map evidence for RHS evaluation with stored RHS history."""
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.RHS_EVALUATION_AVAILABLE: DescriptorCoordinate(True),
                field.RHS_HISTORY_AVAILABLE: DescriptorCoordinate(True),
            }
        )
    )


def nordsieck_history_descriptor(stiffness_estimate: float) -> ParameterDescriptor:
    """Return map evidence for a populated Nordsieck state vector."""
    if stiffness_estimate < 0.0:
        raise ValueError("stiffness estimate must be nonnegative")
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.RHS_EVALUATION_AVAILABLE: DescriptorCoordinate(True),
                field.NORDSIECK_HISTORY_AVAILABLE: DescriptorCoordinate(True),
                field.STIFFNESS_ESTIMATE: DescriptorCoordinate(
                    stiffness_estimate, evidence="upper_bound"
                ),
            }
        )
    )


def semilinear_map_descriptor() -> ParameterDescriptor:
    """Return map evidence for ``f(t, u) = L u + N(t, u)``."""
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.EXACT_LINEAR_OPERATOR_AVAILABLE: DescriptorCoordinate(True),
                field.NONLINEAR_RESIDUAL_AVAILABLE: DescriptorCoordinate(True),
            }
        )
    )


def split_map_descriptor() -> ParameterDescriptor:
    """Return map evidence for an explicit/implicit additive split."""
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.EXPLICIT_COMPONENT_AVAILABLE: DescriptorCoordinate(True),
                field.IMPLICIT_COMPONENT_AVAILABLE: DescriptorCoordinate(True),
                field.IMPLICIT_COMPONENT_DERIVATIVE_ORACLE_KIND: DescriptorCoordinate(
                    "jacobian_callback"
                ),
            }
        )
    )


def hamiltonian_map_descriptor() -> ParameterDescriptor:
    """Return map evidence for a separable Hamiltonian partition."""
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.HAMILTONIAN_PARTITION_AVAILABLE: DescriptorCoordinate(True),
                field.SYMPLECTIC_FORM_DEFECT_UPPER_BOUND: DescriptorCoordinate(0.0),
                field.SYMPLECTIC_FORM_INVARIANT_AVAILABLE: DescriptorCoordinate(True),
            }
        )
    )


def composite_map_descriptor(
    component_count: int,
    *,
    symplectic_form_defect_upper_bound: float = float("inf"),
) -> ParameterDescriptor:
    """Return map evidence for an operator-splitting component decomposition."""
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.ADDITIVE_COMPONENT_COUNT: DescriptorCoordinate(component_count),
                field.SYMPLECTIC_FORM_DEFECT_UPPER_BOUND: DescriptorCoordinate(
                    symplectic_form_defect_upper_bound
                ),
                field.SYMPLECTIC_FORM_INVARIANT_AVAILABLE: DescriptorCoordinate(
                    symplectic_form_defect_upper_bound == 0.0
                ),
            }
        )
    )


def composite_map_descriptor_from_rhs(
    rhs: CompositeRHSProtocol,
) -> ParameterDescriptor:
    """Return composition evidence derived from component-flow structure."""
    defect_upper_bound = 0.0
    for component in rhs.components:
        if not isinstance(component, ComponentFlowProtocol):
            defect_upper_bound = float("inf")
            break
        defect_upper_bound += component.symplectic_form_defect_upper_bound
    return composite_map_descriptor(
        len(rhs.components),
        symplectic_form_defect_upper_bound=defect_upper_bound,
    )


def _derivative_oracle_region(owner: type) -> CoverageRegion:
    return CoverageRegion(
        owner,
        (
            MembershipPredicate(
                SolveRelationField.DERIVATIVE_ORACLE_KIND,
                frozenset({"jacobian_callback"}),
            ),
        ),
    )


def _domain_step_margin_regions(
    owner: type,
    predicates: tuple[ComparisonPredicate | MembershipPredicate, ...],
) -> tuple[CoverageRegion, CoverageRegion]:
    field = MapStructureField
    return (
        CoverageRegion(
            owner,
            predicates + (ComparisonPredicate(field.DOMAIN_STEP_MARGIN, "<=", 0.0),),
        ),
        CoverageRegion(
            owner,
            predicates + (ComparisonPredicate(field.DOMAIN_STEP_MARGIN, ">", 0.0),),
        ),
    )


def _adaptive_advance_regions(owner: type) -> tuple[CoverageRegion, CoverageRegion]:
    return _domain_step_margin_regions(
        owner,
        (
            MembershipPredicate(
                SolveRelationField.DERIVATIVE_ORACLE_KIND,
                frozenset({"jacobian_callback"}),
            ),
        ),
    )


def _rhs_evaluation_region(owner: type) -> CoverageRegion:
    field = MapStructureField
    return CoverageRegion(
        owner,
        (
            MembershipPredicate(field.RHS_EVALUATION_AVAILABLE, frozenset({True})),
            MembershipPredicate(field.RHS_HISTORY_AVAILABLE, frozenset({False})),
            MembershipPredicate(field.NORDSIECK_HISTORY_AVAILABLE, frozenset({False})),
        ),
    )


def _unconstrained_rhs_evaluation_regions(
    owner: type,
) -> tuple[CoverageRegion, CoverageRegion]:
    field = MapStructureField
    return _domain_step_margin_regions(
        owner,
        (
            MembershipPredicate(field.RHS_EVALUATION_AVAILABLE, frozenset({True})),
            MembershipPredicate(field.RHS_HISTORY_AVAILABLE, frozenset({False})),
            MembershipPredicate(field.NORDSIECK_HISTORY_AVAILABLE, frozenset({False})),
            MembershipPredicate(field.CONSERVED_LINEAR_FORM_COUNT, frozenset({0})),
        ),
    )


def _rhs_history_region(owner: type) -> CoverageRegion:
    field = MapStructureField
    return CoverageRegion(
        owner,
        (
            MembershipPredicate(field.RHS_EVALUATION_AVAILABLE, frozenset({True})),
            MembershipPredicate(field.RHS_HISTORY_AVAILABLE, frozenset({True})),
            MembershipPredicate(field.NORDSIECK_HISTORY_AVAILABLE, frozenset({False})),
        ),
    )


def _nordsieck_history_regions(owner: type) -> tuple[CoverageRegion, ...]:
    field = MapStructureField
    return (
        CoverageRegion(
            owner,
            (
                MembershipPredicate(
                    field.NORDSIECK_HISTORY_AVAILABLE, frozenset({True})
                ),
                ComparisonPredicate(field.STIFFNESS_ESTIMATE, "<=", 0.5),
            ),
        ),
        CoverageRegion(
            owner,
            (
                MembershipPredicate(
                    field.NORDSIECK_HISTORY_AVAILABLE, frozenset({True})
                ),
                ComparisonPredicate(field.STIFFNESS_ESTIMATE, ">", 1.0),
            ),
        ),
    )


def _semilinear_map_region(owner: type) -> CoverageRegion:
    field = MapStructureField
    return CoverageRegion(
        owner,
        (
            MembershipPredicate(
                field.EXACT_LINEAR_OPERATOR_AVAILABLE, frozenset({True})
            ),
            MembershipPredicate(field.NONLINEAR_RESIDUAL_AVAILABLE, frozenset({True})),
        ),
    )


def _split_map_region(owner: type) -> CoverageRegion:
    field = MapStructureField
    return CoverageRegion(
        owner,
        (
            MembershipPredicate(field.EXPLICIT_COMPONENT_AVAILABLE, frozenset({True})),
            MembershipPredicate(field.IMPLICIT_COMPONENT_AVAILABLE, frozenset({True})),
            MembershipPredicate(
                field.IMPLICIT_COMPONENT_DERIVATIVE_ORACLE_KIND,
                frozenset({"jacobian_callback", "matrix"}),
            ),
        ),
    )


def _hamiltonian_map_region(owner: type) -> CoverageRegion:
    return CoverageRegion(
        owner,
        (
            MembershipPredicate(
                MapStructureField.HAMILTONIAN_PARTITION_AVAILABLE, frozenset({True})
            ),
            MembershipPredicate(
                MapStructureField.SYMPLECTIC_FORM_INVARIANT_AVAILABLE,
                frozenset({True}),
            ),
        ),
    )


def _composite_map_region(owner: type) -> CoverageRegion:
    return CoverageRegion(
        owner,
        (ComparisonPredicate(MapStructureField.ADDITIVE_COMPONENT_COUNT, ">=", 2),),
    )


def _requested_order(request: AlgorithmRequest) -> int:
    assert request.order is not None
    return request.order


def _component_count(request: AlgorithmRequest) -> int:
    assert request.descriptor is not None
    count = request.descriptor.coordinate(MapStructureField.ADDITIVE_COMPONENT_COUNT)
    assert isinstance(count.value, int)
    return count.value


def _nordsieck_corrector_family(request: AlgorithmRequest) -> Literal["adams", "bdf"]:
    assert request.descriptor is not None
    stiffness = request.descriptor.coordinate(MapStructureField.STIFFNESS_ESTIMATE)
    assert isinstance(stiffness.value, int | float)
    if stiffness.value <= 0.5:
        return "adams"
    if stiffness.value > 1.0:
        return "bdf"
    raise ValueError("Nordsieck corrector family is undefined in stiffness transition")


_CAPABILITIES: tuple[TimeIntegrationCapability, ...] = (
    TimeIntegrationCapability(
        "explicit_runge_kutta",
        None,
        "method_family",
        _contract(
            provides=("one_step", "explicit", "runge_kutta"),
        ),
        1,
        6,
        coverage_regions=(_rhs_evaluation_region(RungeKuttaIntegrator),),
        constructor=lambda request: RungeKuttaIntegrator(_requested_order(request)),
    ),
    TimeIntegrationCapability(
        "implicit_runge_kutta",
        None,
        "method_family",
        _contract(
            provides=("one_step", "implicit", "runge_kutta"),
        ),
        1,
        6,
        coverage_regions=(_derivative_oracle_region(ImplicitRungeKuttaIntegrator),),
        constructor=lambda request: ImplicitRungeKuttaIntegrator(
            _requested_order(request)
        ),
    ),
    TimeIntegrationCapability(
        "additive_runge_kutta",
        None,
        "method_family",
        _contract(
            provides=("one_step", "imex", "runge_kutta"),
        ),
        1,
        4,
        coverage_regions=(_split_map_region(AdditiveRungeKuttaIntegrator),),
        constructor=lambda request: AdditiveRungeKuttaIntegrator(
            _requested_order(request)
        ),
    ),
    TimeIntegrationCapability(
        "lawson_runge_kutta",
        None,
        "method_family",
        _contract(
            provides=("one_step", "exponential", "runge_kutta"),
        ),
        1,
        6,
        coverage_regions=(_semilinear_map_region(LawsonRungeKuttaIntegrator),),
        constructor=lambda request: LawsonRungeKuttaIntegrator(
            _requested_order(request)
        ),
    ),
    TimeIntegrationCapability(
        "symplectic_composition",
        None,
        "method_family",
        _contract(
            provides=("one_step", "symplectic", "composition"),
        ),
        1,
        6,
        supported_orders=frozenset({1, 2, 4, 6}),
        coverage_regions=(_hamiltonian_map_region(SymplecticCompositionIntegrator),),
        constructor=lambda request: SymplecticCompositionIntegrator(
            _requested_order(request)
        ),
    ),
    TimeIntegrationCapability(
        "operator_composition",
        None,
        "method_family",
        _contract(
            provides=("one_step", "operator_splitting", "composition"),
        ),
        1,
        6,
        supported_orders=frozenset({1, 2, 4, 6}),
        coverage_regions=(_composite_map_region(CompositionIntegrator),),
        constructor=lambda request: CompositionIntegrator(
            [RungeKuttaIntegrator(1) for _ in range(_component_count(request))],
            order=_requested_order(request),
        ),
    ),
    TimeIntegrationCapability(
        "adaptive_nordsieck",
        None,
        "controller",
        _contract(
            provides=(
                "advance",
                "nordsieck",
                "adaptive_timestep",
                "variable_order",
                "stiffness_switching",
            ),
        ),
        1,
        6,
        coverage_regions=_adaptive_advance_regions(AdaptiveNordsieckController),
    ),
    TimeIntegrationCapability(
        "explicit_multistep",
        None,
        "method_family",
        _contract(
            provides=("one_step", "explicit", "multistep"),
        ),
        1,
        6,
        coverage_regions=(_rhs_history_region(ExplicitMultistepIntegrator),),
        constructor=lambda request: ExplicitMultistepIntegrator.for_order(
            _requested_order(request)
        ),
    ),
    TimeIntegrationCapability(
        "fixed_order_nordsieck",
        None,
        "method_family",
        _contract(
            provides=("one_step", "nordsieck", "fixed_order"),
        ),
        1,
        6,
        coverage_regions=_nordsieck_history_regions(MultistepIntegrator),
        constructor=lambda request: MultistepIntegrator(
            _nordsieck_corrector_family(request),
            _requested_order(request),
        ),
    ),
    TimeIntegrationCapability(
        "generic_integration_driver",
        None,
        "driver",
        _contract(
            provides=("advance", "adaptive_timestep"),
        ),
        1,
        6,
        coverage_regions=_unconstrained_rhs_evaluation_regions(IntegrationDriver),
    ),
    TimeIntegrationCapability(
        "constraint_aware_controller",
        None,
        "controller",
        _contract(
            provides=("advance", "constraint_lifecycle"),
        ),
        1,
        6,
        coverage_regions=(
            CoverageRegion(
                ConstraintAwareController,
                (
                    ComparisonPredicate(
                        MapStructureField.CONSERVED_LINEAR_FORM_COUNT, ">", 0
                    ),
                    ComparisonPredicate(
                        MapStructureField.ALGEBRAIC_CONSTRAINT_COUNT, ">", 0
                    ),
                    ComparisonPredicate(
                        MapStructureField.DOMAIN_STEP_MARGIN, "<=", 0.0
                    ),
                ),
            ),
        ),
    ),
)


TIME_INTEGRATION_REGISTRY = TimeIntegrationRegistry(_CAPABILITIES)


def time_integration_capabilities() -> tuple[TimeIntegrationCapability, ...]:
    """Return declared time-integration algorithm capabilities."""
    return TIME_INTEGRATION_REGISTRY.capabilities


def time_integration_step_map_regions() -> tuple[CoverageRegion, ...]:
    """Return ODE map regions owned by implementations that perform one step."""
    map_fields = frozenset(MapStructureField)
    return tuple(
        region
        for capability in _CAPABILITIES
        for region in capability.coverage_regions
        if region.referenced_fields <= map_fields
        if callable(getattr(region.owner, "step", None))
    )


def time_integration_step_solve_regions() -> tuple[CoverageRegion, ...]:
    """Return solve-relation regions induced by implementations' step solves."""
    solve_fields = frozenset(SolveRelationField)
    return tuple(
        region
        for capability in _CAPABILITIES
        for region in capability.coverage_regions
        if region.referenced_fields <= solve_fields
        if callable(getattr(region.owner, "step_solve_relation_descriptor", None))
    )


def select_time_integrator(
    request: AlgorithmRequest,
) -> TimeIntegrationCapability:
    """Select a time-integration implementation declaration by capability."""
    return TIME_INTEGRATION_REGISTRY.select(request)


__all__ = [
    "AlgorithmStructureContract",
    "derivative_oracle_descriptor",
    "conserved_rhs_evaluation_descriptor",
    "composite_map_descriptor_from_rhs",
    "composite_map_descriptor",
    "hamiltonian_map_descriptor",
    "nordsieck_history_descriptor",
    "rhs_evaluation_descriptor",
    "rhs_step_diagnostics_descriptor",
    "rhs_history_descriptor",
    "select_time_integrator",
    "semilinear_map_descriptor",
    "split_map_descriptor",
    "TimeIntegrationCapability",
    "time_integration_capabilities",
    "time_integration_step_map_regions",
    "time_integration_step_solve_regions",
    "TimeIntegrationRegistry",
    "TIME_INTEGRATION_REGISTRY",
]
