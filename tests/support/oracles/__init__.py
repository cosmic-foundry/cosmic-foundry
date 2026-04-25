"""Oracle registry entries.

Each import here has two effects:
  1. loads the concrete class (making it visible to __subclasses__())
  2. registers its oracle in CONVERGENCE_ORACLES

Add one line per concrete convergent class when it is introduced.
"""

from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from tests.support.convergence_registry import CONVERGENCE_ORACLES
from tests.support.oracles.diffusive_flux import DiffusiveFluxOracle

CONVERGENCE_ORACLES[DiffusiveFlux] = DiffusiveFluxOracle()
