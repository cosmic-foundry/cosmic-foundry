"""Boundary condition claims.

Claim types:
  _NeumannNullSpaceClaim(ndim, order)
      — DiffusiveFlux with NeumannGhostCells has constant functions in its null
        space: A·ones = 0.  Fails with DirichletGhostCells (odd reflection gives
        a spurious −2c/h² contribution at every boundary cell).

  _ConstantFieldNullLaplacianClaim(ndim)
      — For a constant field φ = c and InhomogeneousDirichletGhostCells(g=c),
        the discretized Laplacian is zero at every cell.  With DirichletGhostCells
        (homogeneous, g=0) the ghost cell is −c, not c, producing a nonzero flux
        2c/h at every boundary face and a nonzero Laplacian at boundary cells.

Both claims are Claim (no calibration) and complete in well under 1s total.
"""

from __future__ import annotations

import pytest
import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.physics.diffusive_flux import DiffusiveFlux
from cosmic_foundry.theory.discrete import (
    DirichletGhostCells,
    DivergenceFormDiscretization,
    InhomogeneousDirichletGhostCells,
    NeumannGhostCells,
)
from cosmic_foundry.theory.discrete.discrete_field import _CallableDiscreteField
from tests.calibration import _NP_BACKEND
from tests.claims import Claim, assemble_linear_op

# Cells per axis for each ndim: keeps total cell count small enough for fast
# SymPy assembly while giving at least one interior cell in every dimension.
_N_PER_AXIS = {1: 8, 2: 4, 3: 3}


class _NeumannNullSpaceClaim(Claim):
    """Claim: DiffusiveFlux with NeumannGhostCells maps the ones vector to zero.

    For a Laplacian with homogeneous Neumann BCs, constant functions are in the
    null space: every face flux is zero (no normal gradient), so the divergence
    is zero at every cell.  The claim assembles the stiffness matrix and verifies
    A·ones = 0 to within floating-point precision.

    This distinguishes NeumannGhostCells (even reflection) from DirichletGhostCells
    (odd reflection): with Dirichlet, ghost = −1 for a ones field, producing a
    nonzero boundary flux of 2/h and a nonzero residual of −2/h² at every
    boundary cell.

    Also verifies n_ghost_layers(order) == order // 2 for each order.
    """

    def __init__(self, ndim: int, order: int) -> None:
        self._ndim = ndim
        self._order = order

    @property
    def description(self) -> str:
        return f"NeumannGhostCells/null_space/{self._ndim}D/order{self._order}"

    def check(self) -> None:
        ndim, order = self._ndim, self._order
        n = _N_PER_AXIS[ndim]
        n_cells = n**ndim

        bc = NeumannGhostCells()
        assert bc.n_ghost_layers(order) == order // 2, (
            f"n_ghost_layers({order}) should be {order // 2}, "
            f"got {bc.n_ghost_layers(order)}"
        )

        manifold = EuclideanManifold(ndim)
        mesh = CartesianMesh(
            origin=tuple(sympy.Rational(0) for _ in range(ndim)),
            spacing=tuple(sympy.Rational(1, n) for _ in range(ndim)),
            shape=(n,) * ndim,
        )
        disc = DivergenceFormDiscretization(DiffusiveFlux(order, manifold), bc)
        op = assemble_linear_op(disc, mesh)

        ones = Tensor([1.0] * n_cells, backend=_NP_BACKEND)
        result = _NP_BACKEND.flatten(op.apply(ones)._value)
        max_abs = max(abs(float(result[i])) for i in range(n_cells))
        assert (
            max_abs < 1e-10
        ), f"A·ones not zero for NeumannGhostCells: max|entry| = {max_abs:.2e}"


class _ConstantFieldNullLaplacianClaim(Claim):
    """Claim: discretized Laplacian of a constant field with matching BC is zero.

    For φ = c (constant) and InhomogeneousDirichletGhostCells(g=c), the ghost
    cell at every boundary equals 2c − c = c.  Every face flux is (c − c)/h = 0,
    so the divergence is zero at every cell.

    With DirichletGhostCells (homogeneous, g=0), the ghost cell is −c, giving a
    boundary face flux of (c − (−c))/h = 2c/h and a Laplacian of −2c/h² ≠ 0 at
    boundary cells for any c ≠ 0.  The claim uses c = 3 to make this failure
    unambiguous.

    Also verifies n_ghost_layers(order) == order // 2 for the new BC.
    """

    def __init__(self, ndim: int) -> None:
        self._ndim = ndim

    @property
    def description(self) -> str:
        return f"InhomogeneousDirichletGhostCells/null_laplacian/{self._ndim}D"

    def check(self) -> None:
        ndim = self._ndim
        n = _N_PER_AXIS[ndim]
        c = sympy.Rational(3)
        order = DiffusiveFlux.min_order

        bc = InhomogeneousDirichletGhostCells(c)
        assert bc.n_ghost_layers(order) == order // 2, (
            f"n_ghost_layers({order}) should be {order // 2}, "
            f"got {bc.n_ghost_layers(order)}"
        )

        manifold = EuclideanManifold(ndim)
        mesh = CartesianMesh(
            origin=tuple(sympy.Rational(0) for _ in range(ndim)),
            spacing=tuple(sympy.Rational(1, n) for _ in range(ndim)),
            shape=(n,) * ndim,
        )
        disc = DivergenceFormDiscretization(
            DiffusiveFlux(order, manifold),
            bc,
        )

        # Constant field: every cell has value c.
        field = _CallableDiscreteField(mesh, lambda _idx: c)
        result = disc(field)

        # Laplacian of a constant function with matching BC is exactly zero.
        from itertools import product as iproduct

        for idx in iproduct(*[range(n) for _ in range(ndim)]):
            val = result(idx)
            assert val == 0, (
                f"Δφ ≠ 0 at cell {idx}: {val} "
                f"(expected 0; check ghost-cell formula 2g−u)"
            )

        # Confirm that homogeneous DirichletGhostCells gives a nonzero answer at
        # the boundary cell — this is what InhomogeneousDirichletGhostCells fixes.
        disc_homog = DivergenceFormDiscretization(
            DiffusiveFlux(order, manifold),
            DirichletGhostCells(),
        )
        result_homog = disc_homog(field)
        h = float(mesh._spacing[0])
        # At the corner cell (0,…,0) all ndim axes are boundary axes.  Each
        # axis contributes a ghost flux of 2c/h at its boundary face; after
        # dividing by cell volume (h^ndim) and multiplying by face_area (h^{ndim-1})
        # each axis contributes +2c/h² to the Laplacian.
        expected_boundary_error = float(ndim * 2 * c / h**2)
        boundary_cell = (0,) * ndim
        val_homog = float(result_homog(boundary_cell))
        assert abs(val_homog - expected_boundary_error) < 1e-6 * abs(
            expected_boundary_error
        ), (
            f"DirichletGhostCells at boundary should give "
            f"{expected_boundary_error:.3f}, got {val_homog:.3f}"
        )


_CLAIMS: list[Claim] = []

for _ndim in [1, 2, 3]:
    for _order in [
        DiffusiveFlux.min_order,
        DiffusiveFlux.min_order + DiffusiveFlux.order_step,
    ]:
        _CLAIMS.append(_NeumannNullSpaceClaim(_ndim, _order))
    _CLAIMS.append(_ConstantFieldNullLaplacianClaim(_ndim))


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_boundary_conditions(claim: Claim) -> None:
    claim.check()
