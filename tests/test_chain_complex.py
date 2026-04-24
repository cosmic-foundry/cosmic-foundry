"""Lane C: verify ∂_{k-1} ∘ ∂_k = 0 on CartesianMesh for n ∈ {1, 2, 3}.

The algebraic identity ∂² = 0 is the discrete counterpart of the
fact that the boundary of a boundary is empty — the same identity
that underlies the divergence theorem and Stokes' theorem.  It is
what makes CellComplex earn its class: without ∂² = 0, the incidence
matrices are not a chain complex and face-flux assembly has no
conservation guarantee.

Verification strategy:
  For each pair (k, k-1) with k ≥ 2, pick a small representative mesh.
  Apply ∂_k to every k-cell to get a signed sum of (k-1)-cells.
  Apply ∂_{k-1} to each (k-1)-cell and accumulate with the outer sign.
  Assert that every (k-2)-cell coefficient is zero.

The k-cell identifier convention is (active_axes, idx) as documented
in ARCHITECTURE.md: active_axes is the sorted tuple of axes the cell
extends along; idx is the lower-corner position in the full vertex grid.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_k_cells(
    shape: tuple[int, ...], k: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate all k-cells of a Cartesian mesh with the given shape.

    Returns list of (active_axes, idx) pairs.  For each choice of k active
    axes, idx ranges over the valid lower-corner positions:
      - Along axis a ∈ active_axes:  idx[a] ∈ [0, shape[a])
      - Along axis b ∉ active_axes:  idx[b] ∈ [0, shape[b]+1)
    """
    ndim = len(shape)
    cells = []
    for active_axes in combinations(range(ndim), k):
        ranges = []
        for a in range(ndim):
            hi = shape[a] if a in active_axes else shape[a] + 1
            ranges.append(range(hi))
        for idx in product(*ranges):
            cells.append((active_axes, idx))
    return cells


def _boundary_squared_coefficients(mesh: CartesianMesh, k: int) -> dict[tuple, int]:
    """Return the coefficient map for (∂_{k-1} ∘ ∂_k) on all k-cells.

    Keys are (k-2)-cell identifiers (active_axes, idx).
    All values should be zero for ∂² = 0 to hold.
    """
    bk = mesh.boundary(k)
    bk1 = mesh.boundary(k - 1)
    coeffs: dict[tuple, int] = defaultdict(int)
    for cell in _all_k_cells(mesh._shape, k):
        for axes1, idx1, sign1 in bk(cell):
            for axes2, idx2, sign2 in bk1((axes1, idx1)):
                coeffs[(axes2, idx2)] += sign1 * sign2
    return dict(coeffs)


# ---------------------------------------------------------------------------
# n = 1: only ∂_1 exists; ∂² = 0 vacuously (no k ≥ 2)
# Verify instead that ∂_1 maps each edge to exactly two vertices with
# opposite signs — the minimal structural check for a 1-complex.
# ---------------------------------------------------------------------------


def test_1d_boundary_orientation():
    """∂_1 maps each 1-cell to its two endpoints with opposite signs."""
    h = sympy.Symbol("h", positive=True)
    mesh = CartesianMesh(origin=(sympy.Integer(0),), spacing=(h,), shape=(4,))
    b1 = mesh.boundary(1)
    for cell in _all_k_cells((4,), 1):
        faces = b1(cell)
        assert len(faces) == 2
        signs = [sign for _, _, sign in faces]
        assert sum(signs) == 0, f"signs do not cancel for cell {cell}: {signs}"


# ---------------------------------------------------------------------------
# n = 2: verify ∂_1 ∘ ∂_2 = 0
# ---------------------------------------------------------------------------


def test_2d_boundary_squared_zero():
    """∂_1 ∘ ∂_2 = 0 on a 3×2 Cartesian mesh."""
    hx, hy = sympy.symbols("hx hy", positive=True)
    mesh = CartesianMesh(
        origin=(sympy.Integer(0), sympy.Integer(0)),
        spacing=(hx, hy),
        shape=(3, 2),
    )
    coeffs = _boundary_squared_coefficients(mesh, k=2)
    nonzero = {cell: c for cell, c in coeffs.items() if c != 0}
    assert not nonzero, f"∂_1 ∘ ∂_2 ≠ 0: non-zero vertex coefficients found:\n{nonzero}"


# ---------------------------------------------------------------------------
# n = 3: verify ∂_1 ∘ ∂_2 = 0 and ∂_2 ∘ ∂_3 = 0
# ---------------------------------------------------------------------------


def test_3d_boundary_squared_zero_faces():
    """∂_1 ∘ ∂_2 = 0 on a 2×2×2 Cartesian mesh (edges → vertices)."""
    hx, hy, hz = sympy.symbols("hx hy hz", positive=True)
    mesh = CartesianMesh(
        origin=(sympy.Integer(0),) * 3,
        spacing=(hx, hy, hz),
        shape=(2, 2, 2),
    )
    coeffs = _boundary_squared_coefficients(mesh, k=2)
    nonzero = {cell: c for cell, c in coeffs.items() if c != 0}
    assert not nonzero, f"∂_1 ∘ ∂_2 ≠ 0: non-zero vertex coefficients:\n{nonzero}"


def test_3d_boundary_squared_zero_cells():
    """∂_2 ∘ ∂_3 = 0 on a 2×2×2 Cartesian mesh (faces → edges)."""
    hx, hy, hz = sympy.symbols("hx hy hz", positive=True)
    mesh = CartesianMesh(
        origin=(sympy.Integer(0),) * 3,
        spacing=(hx, hy, hz),
        shape=(2, 2, 2),
    )
    coeffs = _boundary_squared_coefficients(mesh, k=3)
    nonzero = {cell: c for cell, c in coeffs.items() if c != 0}
    assert not nonzero, f"∂_2 ∘ ∂_3 ≠ 0: non-zero edge coefficients:\n{nonzero}"
