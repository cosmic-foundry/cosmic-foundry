"""Lane C: SPD verification for the assembled FVM Poisson operator.

For FVMDiscretization(mesh, DiffusiveFlux(order), DirichletBC) on CartesianMesh,
the assembled stiffness matrix A is symmetric positive definite with respect to
the discrete inner product.

Verified symbolically at N = 4 in 1-D and 2-D for DiffusiveFlux(2) and
DiffusiveFlux(4).  No eigenvalue computation; SPD is certified via symmetry
check (A = Aᵀ) and Cholesky factorization (all diagonal pivots > 0).
All linear algebra is hand-rolled — no NumPy linalg, no LAPACK.
"""

from __future__ import annotations

from typing import Any

import pytest
import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.geometry.fvm_discretization import FVMDiscretization
from cosmic_foundry.theory.continuous.dirichlet_bc import DirichletBC


def _make_instance(ndim: int, order: int) -> tuple[Any, CartesianMesh, DirichletBC]:
    manifold = EuclideanManifold(ndim)
    mesh = CartesianMesh(
        origin=tuple(sympy.Integer(0) for _ in range(ndim)),
        spacing=tuple(sympy.Integer(1) for _ in range(ndim)),
        shape=tuple(4 for _ in range(ndim)),
    )
    bc = DirichletBC(manifold)
    fvm = FVMDiscretization(mesh, DiffusiveFlux(order, manifold), bc)
    return fvm, mesh, bc


_CASES = [
    (1, 2),
    (1, 4),
    (2, 2),
    (2, 4),
]


@pytest.mark.parametrize(
    "ndim,order",
    _CASES,
    ids=[f"{ndim}D-order{order}" for ndim, order in _CASES],
)
def test_spd(ndim: int, order: int) -> None:
    fvm, _mesh, _bc = _make_instance(ndim, order)
    A = fvm.assemble_matrix()
    n = A.shape[0]

    assert A == A.T, "assembled matrix is not symmetric"

    L = A.cholesky()
    for i in range(n):
        assert L[i, i] > 0, (
            f"Cholesky pivot L[{i},{i}] = {L[i, i]} ≤ 0 "
            f"(ndim={ndim}, order={order})"
        )
