"""GPS time dilation from the Schwarzschild metric.

Derives the relativistic clock rate correction for GPS satellites from the
Schwarzschild metric tensor and validates it against ICD-GPS-200 Table 20-IV.
"""

# %% SchwarzschildManifold
import sympy

from cosmic_foundry.geometry.schwarzschild_manifold import (
    M,
    SchwarzschildManifold,
    r,
    theta,
)
from validation.schwarzschild.figures import time_dilation_figure

spacetime = SchwarzschildManifold()
spacetime.metric.as_matrix()

# %% Satellite clock: proper time rate on a circular geodesic
g_tt = spacetime.metric.component(0, 0).expr
g_phiphi = spacetime.metric.component(3, 3).expr.subs(theta, sympy.pi / 2)

sympy.simplify(-g_tt - g_phiphi * M / r**3)

# %% Ground clock: proper time rate including Earth's rotation
r_E = sympy.Symbol("r_E", positive=True)
Omega = sympy.Symbol("Omega", positive=True)

sympy.simplify((-g_tt - g_phiphi * Omega**2).subs(r, r_E))

# %% time_dilation_figure
time_dilation_figure()
