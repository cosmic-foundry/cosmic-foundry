"""Physics layer: PDE models and concrete simulation state.

physics/ sits above geometry/ and theory/, providing:
  - Concrete NumericalFlux implementations (diffusive, advective,
    advection-diffusion) that combine CartesianMesh geometry with
    DiscreteField arithmetic
  - FVMDiscretization: assembles DiscreteOperator from NumericalFlux
  - State: the concrete Tensor-backed DiscreteField[float] used as
    simulation state throughout time integration

Boundary rule: may import from theory/, geometry/, and computation/.
"""

from __future__ import annotations

from cosmic_foundry.physics.advection_diffusion_flux import AdvectionDiffusionFlux
from cosmic_foundry.physics.advective_flux import AdvectiveFlux
from cosmic_foundry.physics.diffusive_flux import DiffusiveFlux
from cosmic_foundry.physics.fvm_discretization import FVMDiscretization
from cosmic_foundry.physics.operator import Operator
from cosmic_foundry.physics.state import State

__all__ = [
    "AdvectionDiffusionFlux",
    "AdvectiveFlux",
    "DiffusiveFlux",
    "FVMDiscretization",
    "Operator",
    "State",
]
