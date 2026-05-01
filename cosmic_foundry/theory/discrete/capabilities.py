"""Algorithm structure contracts for discrete-operator selection."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmCapability,
    AlgorithmRegistry,
    AlgorithmRequest,
    AlgorithmStructureContract,
)

DiscreteOperatorCapability = AlgorithmCapability
DiscreteOperatorRegistry = AlgorithmRegistry
DiscreteOperatorRequest = AlgorithmRequest


def _contract(
    *,
    requires: tuple[str, ...],
    provides: tuple[str, ...],
) -> AlgorithmStructureContract:
    return AlgorithmStructureContract(frozenset(requires), frozenset(provides))


_CAPABILITIES: tuple[DiscreteOperatorCapability, ...] = (
    DiscreteOperatorCapability(
        "advective_flux",
        "AdvectiveFlux",
        "numerical_flux",
        _contract(
            requires=("cartesian_mesh", "cell_average_field", "smooth_scalar_field"),
            provides=(
                "numerical_flux",
                "advective",
                "centered_stencil",
                "finite_volume",
                "symbolic_stencil",
            ),
        ),
        min_order=2,
        order_step=2,
    ),
    DiscreteOperatorCapability(
        "diffusive_flux",
        "DiffusiveFlux",
        "numerical_flux",
        _contract(
            requires=("cartesian_mesh", "cell_average_field", "smooth_scalar_field"),
            provides=(
                "numerical_flux",
                "diffusive",
                "antisymmetric_stencil",
                "finite_volume",
                "symbolic_stencil",
            ),
        ),
        min_order=2,
        order_step=2,
    ),
    DiscreteOperatorCapability(
        "advection_diffusion_flux",
        "AdvectionDiffusionFlux",
        "numerical_flux",
        _contract(
            requires=("cartesian_mesh", "cell_average_field", "smooth_scalar_field"),
            provides=(
                "numerical_flux",
                "advective",
                "diffusive",
                "finite_volume",
                "symbolic_stencil",
            ),
        ),
        min_order=2,
        order_step=2,
    ),
    DiscreteOperatorCapability(
        "divergence_form_discretization",
        "DivergenceFormDiscretization",
        "discretization",
        _contract(
            requires=(
                "cartesian_mesh",
                "numerical_flux",
                "discrete_boundary_condition",
            ),
            provides=(
                "discrete_operator",
                "divergence_form",
                "finite_volume",
                "boundary_aware",
                "cell_average_field",
            ),
        ),
    ),
)


DISCRETE_OPERATOR_REGISTRY = DiscreteOperatorRegistry(_CAPABILITIES)


def discrete_operator_capabilities() -> tuple[DiscreteOperatorCapability, ...]:
    """Return declared discrete-operator capabilities."""
    return DISCRETE_OPERATOR_REGISTRY.capabilities


def select_discrete_operator(
    request: DiscreteOperatorRequest,
) -> DiscreteOperatorCapability:
    """Select a discrete-operator implementation declaration by capability."""
    return DISCRETE_OPERATOR_REGISTRY.select(request)


__all__ = [
    "DISCRETE_OPERATOR_REGISTRY",
    "DiscreteOperatorCapability",
    "discrete_operator_capabilities",
    "DiscreteOperatorRegistry",
    "DiscreteOperatorRequest",
    "select_discrete_operator",
]
