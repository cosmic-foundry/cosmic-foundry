"""Boundary condition claims.

Every DiscreteBoundaryCondition is characterized by one fact: given a constant
field φ = c, what ghost value does it produce at an out-of-bounds cell?  From
that single number the Laplacian at a boundary cell follows analytically:

    Δφ|_corner = ndim * (c − ghost) / h²

A BC that preserves constants (Neumann, Periodic, InhomogeneousDirichlet with
g=c) produces ghost=c so the Laplacian is zero.  One that does not (Dirichlet
with g=0, ZeroGhostCells) produces a nonzero ghost and a predictable error.

_ConstantFieldClaim tests all three facts for every BC in one uniform shape:
  1. n_ghost_layers(order) == order // 2.
  2. bc.extend(c_field, mesh)((-1,) + (0,)*(ndim-1)) == expected_low_ghost.
  3. disc(c_field)((0,)*ndim) == ndim * (c − expected_low_ghost) / h²  (exact).

This yields 5 BC specs × 3 ndims = 15 claims, all sharing the same structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.discrete import (
    DiffusiveFlux,
    DirichletGhostCells,
    DiscreteBoundaryCondition,
    DivergenceFormDiscretization,
    InhomogeneousDirichletGhostCells,
    NeumannGhostCells,
    PeriodicGhostCells,
    ZeroGhostCells,
)
from cosmic_foundry.theory.discrete.discrete_field import _CallableDiscreteField
from tests.claims import Claim

# Nonzero constant field value — makes failures unambiguous.
_C = sympy.Rational(3)

# Cells per axis: enough for at least one interior cell in every dimension.
_N_PER_AXIS = {1: 8, 2: 4, 3: 3}


@dataclass(frozen=True)
class _BCSpec:
    """Characterizes one DiscreteBoundaryCondition by its constant-field ghost value.

    expected_low_ghost is the value bc.extend(c_field, mesh) returns for the
    ghost index (-1, 0, …, 0) when the field is the constant _C.  All other
    properties (Laplacian factor, null-space membership) follow from this.
    """

    bc: DiscreteBoundaryCondition
    label: str
    expected_low_ghost: Any  # SymPy expression; evaluated at field value _C


_BC_SPECS: list[_BCSpec] = [
    _BCSpec(
        bc=NeumannGhostCells(),
        label="NeumannGhostCells",
        expected_low_ghost=_C,
        # even reflection: ghost = mirror cell = c  →  Δφ = 0
    ),
    _BCSpec(
        bc=PeriodicGhostCells(),
        label="PeriodicGhostCells",
        expected_low_ghost=_C,
        # wraps to field[N-1] = c for a constant field  →  Δφ = 0
    ),
    _BCSpec(
        bc=InhomogeneousDirichletGhostCells(_C),
        label="InhomogeneousDirichletGhostCells",
        expected_low_ghost=_C,
        # u_ghost = 2g − mirror = 2c − c = c  →  Δφ = 0
    ),
    _BCSpec(
        bc=DirichletGhostCells(),
        label="DirichletGhostCells",
        expected_low_ghost=-_C,
        # odd reflection: ghost = −mirror = −c  →  Δφ = 2c/h² per boundary face
    ),
    _BCSpec(
        bc=ZeroGhostCells(),
        label="ZeroGhostCells",
        expected_low_ghost=sympy.Integer(0),
        # absorbing: ghost = 0  →  Δφ = c/h² per boundary face
    ),
]


class _ConstantFieldClaim(Claim[None]):
    """Claim: a BC's ghost formula and its Laplacian consequence are consistent.

    For the constant field φ = _C and a given _BCSpec, three properties are
    checked using exact SymPy arithmetic:
      1. n_ghost_layers(order) == order // 2.
      2. The low-boundary ghost value equals spec.expected_low_ghost.
      3. The Laplacian at the corner cell (0, …, 0) equals
             ndim × (c − expected_low_ghost) / h²
         where h is the cell spacing and ndim is the number of dimensions.
    """

    def __init__(self, spec: _BCSpec, ndim: int) -> None:
        self._spec = spec
        self._ndim = ndim

    @property
    def description(self) -> str:
        return f"constant_field/{self._spec.label}/{self._ndim}D"

    def check(self, _calibration: None) -> None:
        spec = self._spec
        ndim = self._ndim
        n = _N_PER_AXIS[ndim]
        order = DiffusiveFlux.min_order
        h = sympy.Rational(1, n)

        # --- 1. n_ghost_layers ---
        assert spec.bc.n_ghost_layers(order) == order // 2, (
            f"{spec.label}.n_ghost_layers({order}) = "
            f"{spec.bc.n_ghost_layers(order)}, expected {order // 2}"
        )

        manifold = EuclideanManifold(ndim)
        mesh = CartesianMesh(
            origin=tuple(sympy.Rational(0) for _ in range(ndim)),
            spacing=tuple(h for _ in range(ndim)),
            shape=(n,) * ndim,
        )
        c_field = _CallableDiscreteField(mesh, lambda _idx: _C)
        extended = spec.bc.extend(c_field, mesh)

        # --- 2. Ghost value at the low-boundary index (-1, 0, …, 0) ---
        ghost_idx = (-1,) + (0,) * (ndim - 1)
        ghost_val = extended(ghost_idx)
        assert ghost_val == spec.expected_low_ghost, (
            f"{spec.label}/{ndim}D: ghost at {ghost_idx} = {ghost_val}, "
            f"expected {spec.expected_low_ghost}"
        )

        # --- 3. Laplacian at corner cell (0, …, 0) ---
        # Each of the ndim boundary faces contributes (c − ghost)/h² to the
        # Laplacian.  All arithmetic stays in exact SymPy rationals.
        disc = DivergenceFormDiscretization(DiffusiveFlux(order, manifold), spec.bc)
        result = disc(c_field)
        expected_corner = ndim * (_C - spec.expected_low_ghost) / h**2
        assert result((0,) * ndim) == expected_corner, (
            f"{spec.label}/{ndim}D: Δφ at corner = {result((0,)*ndim)}, "
            f"expected {expected_corner}"
        )


_CLAIMS: list[Claim[None]] = [
    _ConstantFieldClaim(spec, ndim) for spec in _BC_SPECS for ndim in [1, 2, 3]
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_boundary_conditions(claim: Claim[None]) -> None:
    claim.check(None)
