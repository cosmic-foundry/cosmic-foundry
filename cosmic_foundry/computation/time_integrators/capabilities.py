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
from cosmic_foundry.computation.time_integrators.implicit import (
    ImplicitRungeKuttaIntegrator,
)

TimeIntegrationCapability = AlgorithmCapability
TimeIntegrationRegistry = AlgorithmRegistry
TimeIntegrationRequest = AlgorithmRequest


def _contract(
    *,
    requires: tuple[str, ...],
    provides: tuple[str, ...],
) -> AlgorithmStructureContract:
    return AlgorithmStructureContract(frozenset(requires), frozenset(provides))


def derivative_oracle_descriptor() -> ParameterDescriptor:
    """Return map evidence for an available Jacobian callback."""
    return ParameterDescriptor(
        {
            SolveRelationField.DERIVATIVE_ORACLE_KIND: DescriptorCoordinate(
                "jacobian_callback"
            )
        }
    )


def _derivative_oracle_region(owner: type) -> CoverageRegion:
    return CoverageRegion(
        owner,
        (
            MembershipPredicate(
                SolveRelationField.DERIVATIVE_ORACLE_KIND,
                frozenset({"jacobian_callback", "matrix"}),
            ),
        ),
    )


_CAPABILITIES: tuple[TimeIntegrationCapability, ...] = (
    TimeIntegrationCapability(
        "explicit_runge_kutta",
        "RungeKuttaIntegrator",
        "method_family",
        _contract(
            requires=("plain_rhs",),
            provides=("one_step", "explicit", "runge_kutta"),
        ),
        1,
        6,
        priority=60,
    ),
    TimeIntegrationCapability(
        "implicit_runge_kutta",
        "ImplicitRungeKuttaIntegrator",
        "method_family",
        _contract(
            requires=(),
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
            requires=("split_rhs",),
            provides=("one_step", "imex", "runge_kutta"),
        ),
        1,
        4,
    ),
    TimeIntegrationCapability(
        "lawson_runge_kutta",
        "LawsonRungeKuttaIntegrator",
        "method_family",
        _contract(
            requires=("semilinear_rhs",),
            provides=("one_step", "exponential", "runge_kutta"),
        ),
        1,
        6,
    ),
    TimeIntegrationCapability(
        "symplectic_composition",
        "SymplecticCompositionIntegrator",
        "method_family",
        _contract(
            requires=("hamiltonian_rhs",),
            provides=("one_step", "symplectic", "composition"),
        ),
        1,
        6,
        supported_orders=frozenset({1, 2, 4, 6}),
    ),
    TimeIntegrationCapability(
        "operator_composition",
        "CompositionIntegrator",
        "method_family",
        _contract(
            requires=("composite_rhs",),
            provides=("one_step", "operator_splitting", "composition"),
        ),
        1,
        6,
        supported_orders=frozenset({1, 2, 4, 6}),
    ),
    TimeIntegrationCapability(
        "explicit_multistep",
        "ExplicitMultistepIntegrator",
        "method_family",
        _contract(
            requires=("plain_rhs",),
            provides=("one_step", "explicit", "multistep"),
        ),
        1,
        6,
        priority=50,
    ),
    TimeIntegrationCapability(
        "fixed_order_nordsieck",
        "MultistepIntegrator",
        "method_family",
        _contract(
            requires=("plain_rhs",),
            provides=("one_step", "nordsieck", "fixed_order"),
        ),
        1,
        6,
    ),
    TimeIntegrationCapability(
        "adaptive_nordsieck",
        "AdaptiveNordsieckController",
        "controller",
        _contract(
            requires=("state_domain",),
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
        "generic_integration_driver",
        "IntegrationDriver",
        "driver",
        _contract(
            requires=("plain_rhs", "time_integrator", "controller"),
            provides=("advance", "adaptive_timestep", "domain_aware_acceptance"),
        ),
        1,
        6,
    ),
    TimeIntegrationCapability(
        "constraint_aware_controller",
        "ConstraintAwareController",
        "controller",
        _contract(
            requires=(),
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


def select_time_integrator(
    request: TimeIntegrationRequest,
) -> TimeIntegrationCapability:
    """Select a time-integration implementation declaration by capability."""
    return TIME_INTEGRATION_REGISTRY.select(request)


__all__ = [
    "AlgorithmStructureContract",
    "derivative_oracle_descriptor",
    "select_time_integrator",
    "TimeIntegrationCapability",
    "time_integration_capabilities",
    "TimeIntegrationRegistry",
    "TimeIntegrationRequest",
    "TIME_INTEGRATION_REGISTRY",
]
