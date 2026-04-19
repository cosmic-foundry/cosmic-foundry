"""Second-order 7-point Laplacian stencil: derivation via Taylor expansion.

Derives the finite-difference weights for the isotropic 7-point Laplacian
stencil in three spatial dimensions from first principles. All load-bearing
algebraic steps are verified by SymPy assertions that execute on import.

Lane C — first-principles origination.
Reference: standard finite-difference theory (Taylor series).

Summary of result
-----------------
For a smooth field φ on a uniform grid with spacing h, the 7-point stencil

    L_h[φ]_{i,j,k} = φ_{i+1,j,k} + φ_{i-1,j,k}
                    + φ_{i,j+1,k} + φ_{i,j-1,k}
                    + φ_{i,j,k+1} + φ_{i,j,k-1}
                    - 6 φ_{i,j,k}

satisfies

    L_h[φ]_{i,j,k} / h² = ∇²φ(x,y,z) + O(h²)

with leading truncation error (h²/12)(∂⁴φ/∂x⁴ + ∂⁴φ/∂y⁴ + ∂⁴φ/∂z⁴).
"""

from __future__ import annotations

import sympy as sp
from sympy.calculus.finite_diff import finite_diff_weights as _fdw

# ---------------------------------------------------------------------------
# Step 1: 1-D Taylor expansion of f(x ± h) in powers of h
# ---------------------------------------------------------------------------
# Expand f(x+h) and f(x-h) as formal power series in h about h=0.
# SymPy's series() treats h as the expansion variable; Derivative objects
# stand in for f′, f″, … at the evaluation point x.

h = sp.Symbol("h", positive=True)
x = sp.Symbol("x", real=True)
f = sp.Function("f")

_ORDER = 6  # expand to O(h^6) so the O(h^4) truncation term is visible

_f_plus = f(x + h).series(h, 0, _ORDER)  # f(x+h) = Σ hⁿ/n! f^(n)(x)
_f_minus = f(x - h).series(h, 0, _ORDER)  # f(x-h) = Σ (-h)ⁿ/n! f^(n)(x)

# ---------------------------------------------------------------------------
# Step 2: Form the three-point stencil combination
# ---------------------------------------------------------------------------
# Odd powers cancel; only even powers survive.
#
#   f(x+h) + f(x-h) - 2f(x) = h²f″(x) + h⁴/12 f⁴(x) + O(h⁶)
#
# removeO() drops the trailing O() term; sp.cancel then simplifies the ratio.

_stencil_1d = (_f_plus + _f_minus - 2 * f(x)).removeO()
_ratio = sp.cancel(sp.expand(_stencil_1d) / h**2)
# _ratio = f″(x) + h²/12 f⁴(x)

# ---------------------------------------------------------------------------
# Assertion 1: leading term (h→0 limit) is f″(x)
# ---------------------------------------------------------------------------

_leading = _ratio.subs(h, 0)
assert sp.simplify(_leading - f(x).diff(x, 2)) == 0, (
    f"Leading term of (f(x+h)+f(x-h)-2f(x))/h² is {_leading}; "
    f"expected {f(x).diff(x, 2)}"
)

# ---------------------------------------------------------------------------
# Assertion 2: truncation error is O(h²) with coefficient f⁴(x)/12
# ---------------------------------------------------------------------------

_truncation_term = _ratio - _leading
_trunc_coeff = sp.cancel(_truncation_term / h**2).subs(h, 0)
assert (
    sp.simplify(_trunc_coeff - f(x).diff(x, 4) / 12) == 0
), f"Leading truncation coefficient is {_trunc_coeff}; expected f''''(x)/12"

# ---------------------------------------------------------------------------
# Step 3: 1-D stencil weights from finite_diff_weights
# ---------------------------------------------------------------------------
# The stencil f(x+h) + f(x-h) - 2f(x) encodes weights [+1, −2, +1] at
# offsets [−1, 0, +1].  Verify these against SymPy's finite_diff_weights
# oracle, which derives them independently via Vandermonde matrix inversion.

_weights_1d = _fdw(2, [-1, 0, 1], 0)[2][-1]

# Assertion 3: neighbor weight is +1
assert (
    _weights_1d[0] == sp.Rational(1) == _weights_1d[2]
), f"Neighbor weights are {_weights_1d[0]}, {_weights_1d[2]}; expected 1"

# Assertion 4: center weight is −2
assert _weights_1d[1] == sp.Rational(
    -2
), f"Center weight is {_weights_1d[1]}; expected -2"

# ---------------------------------------------------------------------------
# Step 4: Extension to 3-D
# ---------------------------------------------------------------------------
# The continuous Laplacian ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z².
# On a uniform isotropic grid (same spacing h in all directions), the
# second-order approximation of each partial derivative uses the same
# three-point stencil applied along each axis independently.
#
# Summing the three independent 1-D stencils:
#
#   (φ_{i+1,j,k} + φ_{i-1,j,k} - 2φ_{i,j,k}) / h²   ≈ ∂²φ/∂x²
#   (φ_{i,j+1,k} + φ_{i,j-1,k} - 2φ_{i,j,k}) / h²   ≈ ∂²φ/∂y²
#   (φ_{i,j,k+1} + φ_{i,j,k-1} - 2φ_{i,j,k}) / h²   ≈ ∂²φ/∂z²
#
# Adding:
#   (sum of 6 face neighbors − 6φ_{i,j,k}) / h²  ≈ ∇²φ

# Assertion 5: center weight in 3-D is 3 × (−2) = −6
_center_weight_3d = 3 * int(_weights_1d[1])
assert _center_weight_3d == -6, f"3-D center weight is {_center_weight_3d}; expected -6"

# Assertion 6: each of the 6 face-neighbor weights is 1
_neighbor_weight = int(_weights_1d[0])
assert _neighbor_weight == 1, f"Face-neighbor weight is {_neighbor_weight}; expected 1"

# ---------------------------------------------------------------------------
# Public constants — derived values, not typed by hand
# ---------------------------------------------------------------------------
# These are the outputs of the SymPy computation above, cast to plain Python
# ints so callers pay no SymPy cost at import time.

#: Center weight of the 7-point Laplacian stencil (3 × 1D center = −6).
CENTER_WEIGHT: int = _center_weight_3d

#: Weight of each of the 6 face neighbors (+1 in every axis direction).
NEIGHBOR_WEIGHT: int = _neighbor_weight

#: Formal approximation order in grid spacing h.
APPROXIMATION_ORDER: int = 2


# ---------------------------------------------------------------------------
# Kernel source generator
# ---------------------------------------------------------------------------


def _coefficients_hash() -> str:
    import hashlib

    canonical = (
        f"CENTER_WEIGHT={CENTER_WEIGHT};"
        f"NEIGHBOR_WEIGHT={NEIGHBOR_WEIGHT};"
        f"APPROXIMATION_ORDER={APPROXIMATION_ORDER}"
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def generate_kernel_source() -> str:
    """Return the source of cosmic_foundry/computation/laplacian.py.

    The returned string embeds a hash of the derived coefficients so that
    the generated file can verify its own integrity at import time.
    """
    digest = _coefficients_hash()
    return f'''\
"""7-point Laplacian stencil kernel.

Generated by derivations/laplacian_stencil.py. Do not edit by hand.
Regenerate with: python scripts/generate_kernels.py

Function:
    domain   — (fields: Array of field arrays on Ω_h ⊆ ℤ³, extent: Extent)
    codomain — un-divided stencil sum L_h[φ] on the interior of Ω_h
    operator — seven-point finite-difference Laplacian, Θ = (radii=(1,1,1)), p = 2
"""

from __future__ import annotations

import hashlib
from typing import Any

from cosmic_foundry.computation.stencil import Stencil

_COEFFICIENTS_HASH = "{digest}"

CENTER_WEIGHT: int = {CENTER_WEIGHT}
NEIGHBOR_WEIGHT: int = {NEIGHBOR_WEIGHT}
APPROXIMATION_ORDER: int = {APPROXIMATION_ORDER}

_canonical = (
    f"CENTER_WEIGHT={{CENTER_WEIGHT}};"
    f"NEIGHBOR_WEIGHT={{NEIGHBOR_WEIGHT}};"
    f"APPROXIMATION_ORDER={{APPROXIMATION_ORDER}}"
)
assert hashlib.sha256(_canonical.encode()).hexdigest()[:16] == _COEFFICIENTS_HASH, (
    "Kernel coefficients do not match the embedded derivation hash. "
    "Regenerate with: python scripts/generate_kernels.py"
)


def _seven_point_fn(fields: tuple[Any, ...], i: Any, j: Any, k: Any) -> Any:
    phi = fields[0]
    return (
        NEIGHBOR_WEIGHT * phi[i - 1, j, k]
        + NEIGHBOR_WEIGHT * phi[i + 1, j, k]
        + NEIGHBOR_WEIGHT * phi[i, j - 1, k]
        + NEIGHBOR_WEIGHT * phi[i, j + 1, k]
        + NEIGHBOR_WEIGHT * phi[i, j, k - 1]
        + NEIGHBOR_WEIGHT * phi[i, j, k + 1]
        + CENTER_WEIGHT * phi[i, j, k]
    )


seven_point_laplacian = Stencil(fn=_seven_point_fn, radii=(1, 1, 1))
'''
