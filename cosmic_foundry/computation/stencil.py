"""Stencil: concrete parametric pointwise stencil Function.

This module also contains the codegen for the canonical second-order 3D Laplacian
stencil. The pattern (_derive, generate) follows the convention that codegen lives
in the same module as the thing it produces.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.computation._codegen import make_hash
from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import Extent, _checked_bounds
from cosmic_foundry.theory.function import Function


@dataclass(frozen=True)
class Stencil(Function):
    """A pointwise stencil operator parametric over a kernel and radii.

    Function:
        domain   — (fields: Array of field arrays on Ω_h ⊆ ℤⁿ, extent: Extent)
        codomain — an array-valued field on Ω_h^int ⊆ Ω_h
        operator — pointwise application of fn over extent

    ``fn(fields, *index_meshgrids) -> scalar`` is the pointwise kernel;
    ``fields[i]`` accesses the i-th input field. ``radii`` gives the
    stencil half-widths per axis.

    ``execute(fields, extent=...)`` is provided automatically.
    """

    fn: Callable[..., Any]
    radii: tuple[int, ...]

    def execute(self, fields: Array[Any], *, extent: Extent) -> Any:
        _validate_halo_access(extent, self.radii, fields)
        return _make_jit_kernel(self.fn, extent)(*fields.elements)


@functools.lru_cache(maxsize=256)
def _make_jit_kernel(fn: Any, extent: Any) -> Callable[..., Any]:
    @jax.jit
    def apply(*jit_inputs: Any) -> Any:
        indices = _region_indices(extent)
        return fn(jit_inputs, *indices)

    return cast(Callable[..., Any], apply)


def _region_indices(extent: Extent) -> tuple[jax.Array, ...]:
    axes = []
    for axis_slice in extent.slices:
        start, stop = _checked_bounds(axis_slice)
        axes.append(jnp.arange(start, stop))
    return tuple(jnp.meshgrid(*axes, indexing="ij"))


def _validate_halo_access(
    extent: Extent,
    radii: tuple[int, ...],
    fields: Array[Any],
) -> None:
    required = extent.expand(radii)
    for input_array in fields.elements:
        if not hasattr(input_array, "shape"):
            msg = "Function inputs must expose a shape"
            raise TypeError(msg)
        shape = tuple(int(axis_size) for axis_size in input_array.shape)
        if len(shape) < required.ndim:
            msg = "Function input rank is smaller than the extent rank"
            raise ValueError(msg)
        for axis, axis_slice in enumerate(required.slices):
            start, stop = _checked_bounds(axis_slice)
            if start < 0 or stop > shape[axis]:
                msg = "Function extent plus radii exceeds input bounds"
                raise ValueError(msg)


__all__ = ["Stencil", "derive_stencil", "seven_point_laplacian"]


# ---------------------------------------------------------------------------
# Parameterizable stencil derivation — SymPy; only runs when called explicitly
# ---------------------------------------------------------------------------


def derive_stencil(deriv_order: int, approx_order: int, ndim: int) -> dict[str, Any]:
    """Derive exact rational weights for an ndim stencil of given derivative order.

    Uses SymPy finite-difference weights. Computes 1D derivative weights of the
    specified order at the requested approximation order, then extends to ndim by
    axis-sum: each axis contributes independently, and the center point receives
    contributions from all axes. This axis-sum extension is appropriate for
    Laplacian-like operators (sum of nth-order directional derivatives).

    Parameters
    ----------
    deriv_order
        Derivative order (1, 2, 3, ...). The order of the derivative to approximate.
    approx_order
        Approximation order (2, 4, 6, ...). Determines the stencil radius
        r = approx_order // 2 and point set size r + r + 1. Must exceed
        deriv_order - 1 to have enough points.
    ndim
        Number of spatial dimensions (1, 2, 3, ...). Required. Extends 1D weights
        to ndim by axis-sum; for single-axis derivatives, use ndim=1.

    Returns
    -------
    dict with keys:
        "terms" : list[tuple[tuple[int,...], Fraction]]
            (offset_tuple, exact_rational_weight) pairs for each stencil point.
            Sorted by offset tuple.
        "radii" : tuple[int,...]
            Half-widths per axis; length == ndim.
        "deriv_order" : int
            The requested derivative order.
        "approx_order" : int
            The requested approximation order.

    Notes
    -----
    Lane C — first-principles origination using SymPy.
    SymPy is imported only when this function is called, not at module load time.
    """
    import sympy as sp  # type: ignore[import-untyped]
    from sympy.calculus.finite_diff import (  # type: ignore[import-untyped]
        finite_diff_weights as _fdw,
    )

    r = approx_order // 2
    points = list(range(-r, r + 1))

    if len(points) <= deriv_order:
        msg = (
            f"approx_order={approx_order} gives {len(points)} points, "
            f"need > {deriv_order} for deriv_order={deriv_order}"
        )
        raise ValueError(msg)

    _weights_1d = _fdw(deriv_order, points, 0)[deriv_order][-1]

    _weights_1d_exact = [sp.Rational(w) for w in _weights_1d]

    terms_dict: dict[tuple[int, ...], Fraction] = {}

    for axis in range(ndim):
        for point_idx, point_offset in enumerate(points):
            offset_list = [0] * ndim
            offset_list[axis] = point_offset
            offset_tuple = tuple(offset_list)

            weight_sp = _weights_1d_exact[point_idx]
            weight_frac = Fraction(weight_sp.p, weight_sp.q)

            if offset_tuple in terms_dict:
                terms_dict[offset_tuple] += weight_frac
            else:
                terms_dict[offset_tuple] = weight_frac

    terms = sorted(terms_dict.items())
    radii = tuple([r] * ndim)

    return {
        "terms": terms,
        "radii": radii,
        "deriv_order": deriv_order,
        "approx_order": approx_order,
    }


def _derive() -> dict[str, Any]:
    """Derive stencil constants and structure for the 7-point Laplacian.

    Returns a dict with:
    - "constants": dict with CENTER_WEIGHT, NEIGHBOR_WEIGHT, APPROXIMATION_ORDER
    - "stencil": dict with "offsets" (face neighbors), "center", "radii"

    Thin wrapper around derive_laplacian_stencil(order=2, ndim=3) that adds
    Laplacian-specific assertions and reformats the output to match the
    expected interface for generate().

    Lane C — first-principles origination.
    """
    result = derive_stencil(deriv_order=2, approx_order=2, ndim=3)
    terms = dict(result["terms"])

    center = terms[(0, 0, 0)]
    neighbors = {t: w for t, w in terms.items() if t != (0, 0, 0)}

    assert int(center) == -6, f"center weight {center}; expected -6"
    assert all(
        int(w) == 1 for w in neighbors.values()
    ), f"neighbor weights {neighbors}; expected all 1"

    constants = {
        "CENTER_WEIGHT": int(center),
        "NEIGHBOR_WEIGHT": int(next(iter(neighbors.values()))),
        "APPROXIMATION_ORDER": result["approx_order"],
    }

    stencil = {
        "offsets": list(neighbors.keys()),
        "center": (0, 0, 0),
        "radii": result["radii"],
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

    lines = [
        f'_COEFFICIENTS_HASH = "{digest}"\n',
        "\n",
        f"CENTER_WEIGHT: int = {cw}\n",
        f"NEIGHBOR_WEIGHT: int = {nw}\n",
        f"APPROXIMATION_ORDER: int = {ao}\n",
        "\n",
        "\n",
    ]

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

    lines.append(
        f"seven_point_laplacian = Stencil(fn=_seven_point_fn, radii={radii})\n"
    )

    return "".join(lines)


# BEGIN GENERATED — do not edit; regenerate with scripts/generate_kernels.py
_COEFFICIENTS_HASH = "5e2b562629d87ae5"

CENTER_WEIGHT: int = -6
NEIGHBOR_WEIGHT: int = 1
APPROXIMATION_ORDER: int = 2


def _seven_point_fn(fields: tuple[Any, ...], i: Any, j: Any, k: Any) -> Any:
    phi = fields[0]
    return (
        NEIGHBOR_WEIGHT * phi[i - 1, j, k]
        + NEIGHBOR_WEIGHT * phi[i, j - 1, k]
        + NEIGHBOR_WEIGHT * phi[i, j, k - 1]
        + NEIGHBOR_WEIGHT * phi[i, j, k + 1]
        + NEIGHBOR_WEIGHT * phi[i, j + 1, k]
        + NEIGHBOR_WEIGHT * phi[i + 1, j, k]
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
