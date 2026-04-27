"""Physics layer: PDE models and concrete simulation state.

physics/ sits above geometry/ and theory/, providing:
  - Concrete NumericalFlux implementations (diffusive, advective,
    advection-diffusion) that combine CartesianMesh geometry with
    DiscreteField arithmetic
  - Operator: the Tensor-backed materialization of a symbolic DiscreteOperator,
    produced by Operator.assemble(disc(), mesh)
  - State: the concrete Tensor-backed DiscreteField[float] used as
    simulation state throughout time integration

FVMDiscretization lives in theory/discrete/ and is re-exported here for
import convenience.

Boundary rule: may import from theory/, geometry/, and computation/.
"""

from __future__ import annotations

from cosmic_foundry.physics.advection_diffusion_flux import AdvectionDiffusionFlux
from cosmic_foundry.physics.advective_flux import AdvectiveFlux
from cosmic_foundry.physics.diffusive_flux import DiffusiveFlux
from cosmic_foundry.physics.operator import Operator
from cosmic_foundry.physics.state import State
from cosmic_foundry.theory.discrete.fvm_discretization import FVMDiscretization

__all__ = [
    "AdvectionDiffusionFlux",
    "AdvectiveFlux",
    "DiffusiveFlux",
    "FVMDiscretization",
    "Operator",
    "State",
]
