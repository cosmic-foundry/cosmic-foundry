"""Time-integration layer: integrators, steppers, controllers, B-series, symplectic."""

from cosmic_foundry.computation.time_integrators._newton import nonlinear_solve
from cosmic_foundry.computation.time_integrators.auto import AutoIntegrator
from cosmic_foundry.computation.time_integrators.bseries import (
    Tree,
    elementary_weight,
    gamma,
    order,
    sigma,
    trees_up_to_order,
)
from cosmic_foundry.computation.time_integrators.constraint_aware import (
    ConstraintAwareController,
    solve_nse,
)
from cosmic_foundry.computation.time_integrators.explicit_multistep import (
    ExplicitMultistepIntegrator,
)
from cosmic_foundry.computation.time_integrators.exponential import (
    CoxMatthewsETDRK4Integrator,
    LawsonRungeKuttaIntegrator,
    PhiFunction,
    SemilinearRHS,
    SemilinearRHSProtocol,
)
from cosmic_foundry.computation.time_integrators.imex import (
    AdditiveRungeKuttaIntegrator,
    SplitRHS,
    SplitRHSProtocol,
)
from cosmic_foundry.computation.time_integrators.implicit import (
    FiniteDiffJacobianRHS,
    ImplicitRungeKuttaIntegrator,
    JacobianRHS,
    WithJacobianRHSProtocol,
    stability_function,
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
)
from cosmic_foundry.computation.time_integrators.reaction_network import (
    ReactionNetworkRHS,
    project_conserved,
)
from cosmic_foundry.computation.time_integrators.runge_kutta import (
    RungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.splitting import (
    CompositeRHS,
    CompositeRHSProtocol,
    CompositionIntegrator,
)
from cosmic_foundry.computation.time_integrators.stepper import (
    Integrator,
    IntegratorSelectionResult,
)
from cosmic_foundry.computation.time_integrators.stiffness import (
    FamilyName,
    FamilySwitch,
    FamilySwitchingNordsieckIntegrator,
    StiffnessDiagnostic,
    StiffnessSwitcher,
    nordsieck_solution_distance,
)
from cosmic_foundry.computation.time_integrators.symplectic import (
    HamiltonianRHS,
    HamiltonianRHSProtocol,
    SymplecticCompositionIntegrator,
)
from cosmic_foundry.computation.time_integrators.variable_order import (
    OrderDecision,
    OrderSelector,
    VariableOrderNordsieckIntegrator,
)
from cosmic_foundry.computation.time_integrators.vode import VODEController

__all__ = [
    "Tree",
    "AutoIntegrator",
    "BlackBoxRHS",
    "ConstraintAwareController",
    "nonlinear_solve",
    "solve_nse",
    "ReactionNetworkRHS",
    "project_conserved",
    "ConstantStep",
    "Controller",
    "IntegratorSelectionResult",
    "PIController",
    "RHSProtocol",
    "ODEState",
    "RungeKuttaIntegrator",
    "TimeIntegrator",
    "Integrator",
    "elementary_weight",
    "gamma",
    "order",
    "sigma",
    "trees_up_to_order",
    "CoxMatthewsETDRK4Integrator",
    "LawsonRungeKuttaIntegrator",
    "SemilinearRHS",
    "SemilinearRHSProtocol",
    "PhiFunction",
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
    "ExplicitMultistepIntegrator",
    "MultistepIntegrator",
    "NordsieckHistory",
    "OrderDecision",
    "OrderSelector",
    "VariableOrderNordsieckIntegrator",
    "FamilyName",
    "FamilySwitch",
    "FamilySwitchingNordsieckIntegrator",
    "StiffnessDiagnostic",
    "StiffnessSwitcher",
    "nordsieck_solution_distance",
    "VODEController",
]
