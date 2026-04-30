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
    PhiFunction,
    SemilinearRHS,
    SemilinearRHSProtocol,
    cox_matthews_etdrk4,
)
from cosmic_foundry.computation.time_integrators.imex import (
    AdditiveRungeKuttaIntegrator,
    SplitRHS,
    SplitRHSProtocol,
    ars222,
)
from cosmic_foundry.computation.time_integrators.implicit import (
    FiniteDiffJacobianRHS,
    ImplicitRungeKuttaIntegrator,
    JacobianRHS,
    WithJacobianRHSProtocol,
    backward_euler,
    crouzeix_3,
    implicit_midpoint,
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
    bogacki_shampine,
    dormand_prince,
    forward_euler,
    midpoint,
    rk4,
)
from cosmic_foundry.computation.time_integrators.splitting import (
    CompositeRHS,
    CompositeRHSProtocol,
    CompositionIntegrator,
    SplittingStep,
    strang_steps,
    yoshida_steps,
)
from cosmic_foundry.computation.time_integrators.stepper import (
    IntegratorSelectionResult,
    TimeStepper,
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
    "TimeStepper",
    "bogacki_shampine",
    "dormand_prince",
    "elementary_weight",
    "forward_euler",
    "gamma",
    "midpoint",
    "order",
    "rk4",
    "sigma",
    "trees_up_to_order",
    "CoxMatthewsETDRK4Integrator",
    "SemilinearRHS",
    "SemilinearRHSProtocol",
    "PhiFunction",
    "cox_matthews_etdrk4",
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
    "backward_euler",
    "crouzeix_3",
    "implicit_midpoint",
    "stability_function",
    "AdditiveRungeKuttaIntegrator",
    "SplitRHS",
    "SplitRHSProtocol",
    "ars222",
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
    "SplittingStep",
    "strang_steps",
    "yoshida_steps",
]
