"""7-point Laplacian stencil kernel.

Function:
    domain   — (fields: Array of field arrays on Ω_h ⊆ ℤ³, extent: Extent)
    codomain — un-divided stencil sum L_h[φ] on the interior of Ω_h
    operator — seven-point finite-difference Laplacian, Θ = (radii=(1,1,1)), p = 2

The generated block below is produced by calling generate() and writing
the result back into this file. SymPy is not imported at module load time.
To regenerate: python scripts/generate_kernels.py
"""

from __future__ import annotations

from typing import Any

from cosmic_foundry.computation._codegen import make_hash
from cosmic_foundry.computation.stencil import Stencil

# BEGIN GENERATED — do not edit; regenerate with scripts/generate_kernels.py
_COEFFICIENTS_HASH = "5e2b562629d87ae5"

CENTER_WEIGHT: int = -6
NEIGHBOR_WEIGHT: int = 1
APPROXIMATION_ORDER: int = 2


def _seven_point_fn(fields: tuple[Any, ...], i: Any, j: Any, k: Any) -> Any:
    phi = fields[0]
    return (
        NEIGHBOR_WEIGHT * phi[i + 1, j, k]
        + NEIGHBOR_WEIGHT * phi[i - 1, j, k]
        + NEIGHBOR_WEIGHT * phi[i, j + 1, k]
        + NEIGHBOR_WEIGHT * phi[i, j - 1, k]
        + NEIGHBOR_WEIGHT * phi[i, j, k + 1]
        + NEIGHBOR_WEIGHT * phi[i, j, k - 1]
        + CENTER_WEIGHT * phi[i, j, k]
    )


seven_point_laplacian = Stencil(fn=_seven_point_fn, radii=(1, 1, 1))
# END GENERATED

assert (
    make_hash(
        {
            "CENTER_WEIGHT": CENTER_WEIGHT,
            "NEIGHBOR_WEIGHT": NEIGHBOR_WEIGHT,
            "APPROXIMATION_ORDER": APPROXIMATION_ORDER,
        }
    )
    == _COEFFICIENTS_HASH
), (
    "Constants do not match the derivation hash. "
    "Regenerate with: python scripts/generate_kernels.py"
)


# ---------------------------------------------------------------------------
# Derivation — SymPy; only runs when called explicitly via generate()
# ---------------------------------------------------------------------------


def _derive() -> dict:
    """Derive stencil constants and structure via Taylor expansion.

    Returns a dict with:
    - "constants": dict with CENTER_WEIGHT, NEIGHBOR_WEIGHT, APPROXIMATION_ORDER
    - "stencil": dict with "offsets" (face neighbors), "center", "radii"

    Derives the weights for the second-order 7-point Laplacian stencil from
    first principles: Taylor-expands f(x±h), forms f(x+h)+f(x-h)−2f(x),
    and asserts symbolically that the leading term is f″(x) with truncation
    error O(h²). Extends to 3D by summing three independent 1D stencils.

    Lane C — first-principles origination.
    Reference: standard finite-difference theory (Taylor series).
    """
    import sympy as sp  # type: ignore[import-untyped]
    from sympy.calculus.finite_diff import (  # type: ignore[import-untyped]
        finite_diff_weights as _fdw,
    )

    h = sp.Symbol("h", positive=True)
    x = sp.Symbol("x", real=True)
    f = sp.Function("f")

    _ORDER = 6  # expand to O(h^6) so the O(h^4) truncation term is visible

    _f_plus = f(x + h).series(h, 0, _ORDER)
    _f_minus = f(x - h).series(h, 0, _ORDER)

    _stencil_1d = (_f_plus + _f_minus - 2 * f(x)).removeO()
    _ratio = sp.cancel(sp.expand(_stencil_1d) / h**2)

    # Assertion 1: leading term is f″(x)
    _leading = _ratio.subs(h, 0)
    assert (
        sp.simplify(_leading - f(x).diff(x, 2)) == 0
    ), f"Leading term is {_leading}; expected f''(x)"

    # Assertion 2: truncation error is O(h²) with coefficient f⁴(x)/12
    _trunc_coeff = sp.cancel((_ratio - _leading) / h**2).subs(h, 0)
    assert (
        sp.simplify(_trunc_coeff - f(x).diff(x, 4) / 12) == 0
    ), f"Truncation coefficient is {_trunc_coeff}; expected f''''(x)/12"

    # Assertion 3-4: 1D stencil weights via independent Vandermonde derivation
    _weights_1d = _fdw(2, [-1, 0, 1], 0)[2][-1]
    assert _weights_1d[0] == sp.Rational(1) == _weights_1d[2]
    assert _weights_1d[1] == sp.Rational(-2)

    # Assertion 5-6: 3D weights (sum three independent 1D stencils)
    center = 3 * int(_weights_1d[1])
    neighbor = int(_weights_1d[0])
    assert center == -6
    assert neighbor == 1

    constants = {
        "CENTER_WEIGHT": center,
        "NEIGHBOR_WEIGHT": neighbor,
        "APPROXIMATION_ORDER": 2,
    }

    # Stencil structure: six face neighbors at radius 1
    stencil = {
        "offsets": [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ],
        "center": (0, 0, 0),
        "radii": (1, 1, 1),
    }

    return {
        "constants": constants,
        "stencil": stencil,
    }


def generate() -> str:
    """Generate the full kernel block: constants, function, and Stencil instance.

    The generator script (scripts/generate_kernels.py) writes this between
    the BEGIN GENERATED / END GENERATED sentinels in this file.
    """
    result = _derive()
    constants = result["constants"]
    stencil = result["stencil"]
    offsets = stencil["offsets"]
    radii = stencil["radii"]

    digest = make_hash(constants)
    cw = constants["CENTER_WEIGHT"]
    nw = constants["NEIGHBOR_WEIGHT"]
    ao = constants["APPROXIMATION_ORDER"]

    # Header: hash and constant declarations
    lines = [
        f'_COEFFICIENTS_HASH = "{digest}"\n',
        "\n",
        f"CENTER_WEIGHT: int = {cw}\n",
        f"NEIGHBOR_WEIGHT: int = {nw}\n",
        f"APPROXIMATION_ORDER: int = {ao}\n",
        "\n",
        "\n",
    ]

    # Kernel function: iterate over offsets and format each term
    lines.append(
        "def _seven_point_fn(fields: tuple[Any, ...], i: Any, j: Any, k: Any) -> Any:\n"
    )
    lines.append("    phi = fields[0]\n")
    lines.append("    return (\n")

    for i, (di, dj, dk) in enumerate(offsets):
        indices = []
        for idx_name, delta in [("i", di), ("j", dj), ("k", dk)]:
            if delta > 0:
                indices.append(f"{idx_name} + {delta}")
            elif delta < 0:
                indices.append(f"{idx_name} - {abs(delta)}")
            else:
                indices.append(idx_name)
        idx_str = ", ".join(indices)
        if i == 0:
            lines.append(f"        NEIGHBOR_WEIGHT * phi[{idx_str}]\n")
        else:
            lines.append(f"        + NEIGHBOR_WEIGHT * phi[{idx_str}]\n")

    lines.append("        + CENTER_WEIGHT * phi[i, j, k]\n")
    lines.append("    )\n")
    lines.append("\n")
    lines.append("\n")

    # Stencil instantiation
    lines.append(
        f"seven_point_laplacian = Stencil(fn=_seven_point_fn, radii={radii})\n"
    )

    return "".join(lines)
