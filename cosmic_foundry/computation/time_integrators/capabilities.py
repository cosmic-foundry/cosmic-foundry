"""Algorithm structure contracts for time-integration selection."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmCapability,
    AlgorithmRegistry,
    AlgorithmRequest,
    AlgorithmStructureContract,
    ComparisonPredicate,
    CoverageRegion,
    DescriptorCoordinate,
    DescriptorField,
    MapStructureField,
    MembershipPredicate,
    ParameterDescriptor,
    ReactionNetworkField,
    SolveRelationField,
)
from cosmic_foundry.computation.time_integrators.adaptive_nordsieck import (
    AdaptiveNordsieckController,
)
from cosmic_foundry.computation.time_integrators.constraint_aware import (
    ConstraintAwareController,
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
        field.SYMPLECTIC_FORM_INVARIANT_AVAILABLE: DescriptorCoordinate(False),
        field.ADDITIVE_COMPONENT_COUNT: DescriptorCoordinate(0),
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


def nordsieck_history_descriptor() -> ParameterDescriptor:
    """Return map evidence for a populated Nordsieck state vector."""
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.RHS_EVALUATION_AVAILABLE: DescriptorCoordinate(True),
                field.NORDSIECK_HISTORY_AVAILABLE: DescriptorCoordinate(True),
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
                field.SYMPLECTIC_FORM_INVARIANT_AVAILABLE: DescriptorCoordinate(True),
            }
        )
    )


def composite_map_descriptor(
    component_count: int,
    *,
    symplectic_form_invariant_available: bool = False,
) -> ParameterDescriptor:
    """Return map evidence for an operator-splitting component decomposition."""
    field = MapStructureField
    return ParameterDescriptor(
        _map_structure_coordinates(
            {
                field.ADDITIVE_COMPONENT_COUNT: DescriptorCoordinate(component_count),
                field.SYMPLECTIC_FORM_INVARIANT_AVAILABLE: DescriptorCoordinate(
                    symplectic_form_invariant_available
                ),
            }
        )
    )


def composite_map_descriptor_from_rhs(
    rhs: CompositeRHSProtocol,
) -> ParameterDescriptor:
    """Return composition evidence derived from component-flow structure."""
    return composite_map_descriptor(
        len(rhs.components),
        symplectic_form_invariant_available=all(
            isinstance(component, ComponentFlowProtocol)
            and component.preserves_symplectic_form
            for component in rhs.components
        ),
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


def _nordsieck_history_region(owner: type) -> CoverageRegion:
    field = MapStructureField
    return CoverageRegion(
        owner,
        (MembershipPredicate(field.NORDSIECK_HISTORY_AVAILABLE, frozenset({True})),),
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


_CAPABILITIES: tuple[TimeIntegrationCapability, ...] = (
    TimeIntegrationCapability(
        "explicit_runge_kutta",
        "RungeKuttaIntegrator",
        "method_family",
        _contract(
            provides=("one_step", "explicit", "runge_kutta"),
        ),
        1,
        6,
        coverage_regions=(_rhs_evaluation_region(RungeKuttaIntegrator),),
    ),
    TimeIntegrationCapability(
        "implicit_runge_kutta",
        "ImplicitRungeKuttaIntegrator",
        "method_family",
        _contract(
            provides=("one_step", "implicit", "runge_kutta"),
        ),
        1,
        6,
        coverage_regions=(_derivative_oracle_region(ImplicitRungeKuttaIntegrator),),
    ),
    TimeIntegrationCapability(
        "additive_runge_kutta",
        "AdditiveRungeKuttaIntegrator",
        "method_family",
        _contract(
            provides=("one_step", "imex", "runge_kutta"),
        ),
        1,
        4,
        coverage_regions=(_split_map_region(AdditiveRungeKuttaIntegrator),),
    ),
    TimeIntegrationCapability(
        "lawson_runge_kutta",
        "LawsonRungeKuttaIntegrator",
        "method_family",
        _contract(
            provides=("one_step", "exponential", "runge_kutta"),
        ),
        1,
        6,
        coverage_regions=(_semilinear_map_region(LawsonRungeKuttaIntegrator),),
    ),
    TimeIntegrationCapability(
        "symplectic_composition",
        "SymplecticCompositionIntegrator",
        "method_family",
        _contract(
            provides=("one_step", "symplectic", "composition"),
        ),
        1,
        6,
        supported_orders=frozenset({1, 2, 4, 6}),
        coverage_regions=(_hamiltonian_map_region(SymplecticCompositionIntegrator),),
    ),
    TimeIntegrationCapability(
        "operator_composition",
        "CompositionIntegrator",
        "method_family",
        _contract(
            provides=("one_step", "operator_splitting", "composition"),
        ),
        1,
        6,
        supported_orders=frozenset({1, 2, 4, 6}),
        coverage_regions=(_composite_map_region(CompositionIntegrator),),
    ),
    TimeIntegrationCapability(
        "adaptive_nordsieck",
        "AdaptiveNordsieckController",
        "controller",
        _contract(
            provides=(
                "advance",
                "nordsieck",
                "adaptive_timestep",
                "variable_order",
                "stiffness_switching",
                "domain_aware_acceptance",
            ),
        ),
        1,
        6,
        coverage_regions=(_derivative_oracle_region(AdaptiveNordsieckController),),
    ),
    TimeIntegrationCapability(
        "explicit_multistep",
        "ExplicitMultistepIntegrator",
        "method_family",
        _contract(
            provides=("one_step", "explicit", "multistep"),
        ),
        1,
        6,
        coverage_regions=(_rhs_history_region(ExplicitMultistepIntegrator),),
    ),
    TimeIntegrationCapability(
        "fixed_order_nordsieck",
        "MultistepIntegrator",
        "method_family",
        _contract(
            provides=("one_step", "nordsieck", "fixed_order"),
        ),
        1,
        6,
        coverage_regions=(_nordsieck_history_region(MultistepIntegrator),),
    ),
    TimeIntegrationCapability(
        "generic_integration_driver",
        "IntegrationDriver",
        "driver",
        _contract(
            provides=("advance", "adaptive_timestep", "domain_aware_acceptance"),
        ),
        1,
        6,
        coverage_regions=(_rhs_evaluation_region(IntegrationDriver),),
    ),
    TimeIntegrationCapability(
        "constraint_aware_controller",
        "ConstraintAwareController",
        "controller",
        _contract(
            provides=("advance", "constraint_lifecycle", "domain_aware_acceptance"),
        ),
        1,
        6,
        coverage_regions=(
            CoverageRegion(
                ConstraintAwareController,
                (
                    ComparisonPredicate(
                        ReactionNetworkField.CONSERVATION_LAW_COUNT, ">", 0
                    ),
                    ComparisonPredicate(
                        ReactionNetworkField.EQUILIBRIUM_CONSTRAINT_COUNT, ">", 0
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
    "composite_map_descriptor_from_rhs",
    "composite_map_descriptor",
    "hamiltonian_map_descriptor",
    "nordsieck_history_descriptor",
    "rhs_evaluation_descriptor",
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
