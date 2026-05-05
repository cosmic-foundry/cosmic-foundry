"""Time-integration layer: integrators, steppers, controllers, B-series, symplectic."""

from cosmic_foundry.computation.time_integrators.adaptive_nordsieck import (
    AdaptiveNordsieckController,
)
from cosmic_foundry.computation.time_integrators.auto import AutoIntegrator
from cosmic_foundry.computation.time_integrators.bseries import (
    Tree,
    elementary_weight,
    gamma,
    order,
    sigma,
    trees_up_to_order,
)
from cosmic_foundry.computation.time_integrators.capabilities import (
    AlgorithmStructureContract,
    TimeIntegrationCapability,
    TimeIntegrationRegistry,
    select_time_integrator,
    time_integration_capabilities,
)
from cosmic_foundry.computation.time_integrators.constraint_aware import (
    ConstraintAwareController,
    nse_root_relation,
    solve_nse,
)
from cosmic_foundry.computation.time_integrators.domains import (
    DomainCheck,
    DomainViolation,
    NonnegativeStateDomain,
    StateDomain,
    check_state_domain,
    predict_domain_step_limit,
)
from cosmic_foundry.computation.time_integrators.explicit_multistep import (
    ExplicitMultistepIntegrator,
)
from cosmic_foundry.computation.time_integrators.exponential import (
    LawsonRungeKuttaIntegrator,
    PhiFunction,
    SemilinearRHS,
    SemilinearRHSProtocol,
)
from cosmic_foundry.computation.time_integrators.imex import (
    AdditiveRungeKuttaIntegrator,
    SplitRHS,
    SplitRHSProtocol,
    imex_implicit_stage_root_relation,
)
from cosmic_foundry.computation.time_integrators.implicit import (
    FiniteDiffJacobianRHS,
    ImplicitRungeKuttaIntegrator,
    JacobianRHS,
    WithJacobianRHSProtocol,
    stability_function,
)
from cosmic_foundry.computation.time_integrators.integration_driver import (
    IntegrationDriver,
    IntegrationSelectionResult,
)
from cosmic_foundry.computation.time_integrators.integrator import (
    BlackBoxRHS,
    ConstantStep,
    Controller,
    ODEState,
    PIController,
    RHSProtocol,
    TimeIntegrator,
)
from cosmic_foundry.computation.time_integrators.nordsieck import (
    MultistepIntegrator,
    NordsieckHistory,
    adams_corrector_root_relation,
    bdf_corrector_root_relation,
)
from cosmic_foundry.computation.time_integrators.reaction_network import (
    LinearReactionNetworkRHS,
    ReactionNetworkRHS,
    UnitTransferRates,
    UnitTransferTransitionSystemProtocol,
    project_conserved,
)
from cosmic_foundry.computation.time_integrators.runge_kutta import (
    RungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.solve_relation import (
    AffineRHSProtocol,
    DirectionalDerivativeRHSProtocol,
    JacobianRHSProtocol,
    affine_stage_residual_relation,
    dirk_stage_directional_derivative_root_relation,
    dirk_stage_root_relation,
    implicit_stage_directional_derivative_root_relation,
    implicit_stage_root_relation,
    time_integrator_step_solve_relation_descriptor,
)
from cosmic_foundry.computation.time_integrators.splitting import (
    ComponentFlowProtocol,
    ComponentFlowRHS,
    CompositeRHS,
    CompositeRHSProtocol,
    CompositionIntegrator,
)
from cosmic_foundry.computation.time_integrators.stiffness import (
    FamilyName,
    FamilySwitch,
    StiffnessDiagnostic,
    StiffnessSwitcher,
)
from cosmic_foundry.computation.time_integrators.symplectic import (
    HamiltonianRHS,
    HamiltonianRHSProtocol,
    SymplecticCompositionIntegrator,
)
from cosmic_foundry.computation.time_integrators.variable_order import (
    OrderDecision,
    OrderSelector,
)

__all__ = [
    "AlgorithmStructureContract",
    "AffineRHSProtocol",
    "DirectionalDerivativeRHSProtocol",
    "JacobianRHSProtocol",
    "Tree",
    "AutoIntegrator",
    "BlackBoxRHS",
    "check_state_domain",
    "ConstraintAwareController",
    "DomainCheck",
    "DomainViolation",
    "NonnegativeStateDomain",
    "nse_root_relation",
    "predict_domain_step_limit",
    "solve_nse",
    "StateDomain",
    "LinearReactionNetworkRHS",
    "ReactionNetworkRHS",
    "UnitTransferRates",
    "UnitTransferTransitionSystemProtocol",
    "project_conserved",
    "ConstantStep",
    "Controller",
    "IntegrationSelectionResult",
    "PIController",
    "RHSProtocol",
    "ODEState",
    "RungeKuttaIntegrator",
    "select_time_integrator",
    "TimeIntegrator",
    "affine_stage_residual_relation",
    "dirk_stage_directional_derivative_root_relation",
    "dirk_stage_root_relation",
    "implicit_stage_directional_derivative_root_relation",
    "implicit_stage_root_relation",
    "time_integrator_step_solve_relation_descriptor",
    "IntegrationDriver",
    "TimeIntegrationCapability",
    "time_integration_capabilities",
    "TimeIntegrationRegistry",
    "elementary_weight",
    "gamma",
    "order",
    "sigma",
    "trees_up_to_order",
    "LawsonRungeKuttaIntegrator",
    "SemilinearRHS",
    "SemilinearRHSProtocol",
    "PhiFunction",
    "ComponentFlowProtocol",
    "ComponentFlowRHS",
    "CompositeRHS",
    "CompositeRHSProtocol",
    "CompositionIntegrator",
    "HamiltonianRHS",
    "HamiltonianRHSProtocol",
    "SymplecticCompositionIntegrator",
    "FiniteDiffJacobianRHS",
    "JacobianRHS",
    "ImplicitRungeKuttaIntegrator",
    "WithJacobianRHSProtocol",
    "stability_function",
    "AdditiveRungeKuttaIntegrator",
    "SplitRHS",
    "SplitRHSProtocol",
    "imex_implicit_stage_root_relation",
    "ExplicitMultistepIntegrator",
    "MultistepIntegrator",
    "NordsieckHistory",
    "adams_corrector_root_relation",
    "bdf_corrector_root_relation",
    "OrderDecision",
    "OrderSelector",
    "FamilyName",
    "FamilySwitch",
    "StiffnessDiagnostic",
    "StiffnessSwitcher",
    "AdaptiveNordsieckController",
]
