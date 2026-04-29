"""Time-integration layer: integrators, steppers, controllers, and B-series."""

from cosmic_foundry.computation.time_integrators.bseries import (
    Tree,
    elementary_weight,
    gamma,
    order,
    sigma,
    trees_up_to_order,
)
from cosmic_foundry.computation.time_integrators.integrator import (
    BlackBoxRHS,
    ConstantStep,
    Controller,
    PIController,
    RHSProtocol,
    RKState,
    TimeIntegrator,
)
from cosmic_foundry.computation.time_integrators.runge_kutta import (
    RungeKuttaIntegrator,
    bogacki_shampine,
    dormand_prince,
    forward_euler,
    heun,
    midpoint,
    ralston,
    rk4,
)
from cosmic_foundry.computation.time_integrators.stepper import (
    IntegratorSelectionResult,
    TimeStepper,
)

__all__ = [
    "Tree",
    "BlackBoxRHS",
    "ConstantStep",
    "Controller",
    "IntegratorSelectionResult",
    "PIController",
    "RHSProtocol",
    "RKState",
    "RungeKuttaIntegrator",
    "TimeIntegrator",
    "TimeStepper",
    "bogacki_shampine",
    "dormand_prince",
    "elementary_weight",
    "forward_euler",
    "gamma",
    "heun",
    "midpoint",
    "order",
    "ralston",
    "rk4",
    "sigma",
    "trees_up_to_order",
]
