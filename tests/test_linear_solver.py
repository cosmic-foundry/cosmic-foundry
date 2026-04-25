"""Parametric test framework for LinearSolver implementations.

Two claims for every registered case:

  test_linear_solver_solves     — the solver returns u with ‖u − u_exact‖_{L²_h} < atol
  test_jacobi_spectral_radius   — the Jacobi iteration matrix M_J = I − D⁻¹A has
                                   spectral radius < 1, guaranteeing convergence

Adding a new LinearSolver or a new discretization: add to _SOLVER_CASES or
_SPECTRAL_CASES respectively.  No new test functions are needed.
"""

from __future__ import annotations

import math

import pytest
import sympy

from cosmic_foundry.computation.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.geometry.fvm_discretization import FVMDiscretization
from cosmic_foundry.theory.continuous.dirichlet_bc import DirichletBC
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_manifold_1d = EuclideanManifold(1)
_manifold_2d = EuclideanManifold(2)

# 1-D mesh: 4 cells, h = 1/4
_mesh_1d = CartesianMesh(
    origin=(sympy.Integer(0),),
    spacing=(sympy.Rational(1, 4),),
    shape=(4,),
)

# 2-D mesh: 2×2 cells, h = 1/2
_mesh_2d = CartesianMesh(
    origin=(sympy.Integer(0), sympy.Integer(0)),
    spacing=(sympy.Rational(1, 2), sympy.Rational(1, 2)),
    shape=(2, 2),
)

_disc_1d = FVMDiscretization(
    _mesh_1d,
    DiffusiveFlux(DiffusiveFlux.min_order, _manifold_1d),
    DirichletBC(_manifold_1d),
)

_disc_2d = FVMDiscretization(
    _mesh_2d,
    DiffusiveFlux(DiffusiveFlux.min_order, _manifold_2d),
    DirichletBC(_manifold_2d),
)

# ---------------------------------------------------------------------------
# 1-D manufactured pair: u_exact ≡ 1, rhs = Lₕ u_exact
#
# For u = [1,1,1,1] with h = 1/4 and ghost-cell Dirichlet:
#   boundary cell residual = 2/h² = 32
#   interior cell residual = 0
#   → rhs = [32, 0, 0, 32]
# ---------------------------------------------------------------------------

_rhs_1d = LazyMeshFunction(
    _mesh_1d,
    lambda idx: sympy.Integer(32) if idx[0] in (0, 3) else sympy.Integer(0),
)
_exact_1d = LazyMeshFunction(_mesh_1d, lambda _idx: 1.0)

# ---------------------------------------------------------------------------
# 2-D manufactured pair: u_exact ≡ 1, rhs = Lₕ u_exact
#
# For u = [[1,1],[1,1]] with h = 1/2 and ghost-cell Dirichlet:
#   every cell is a corner (2×2 mesh has no pure-interior cells), so each cell
#   has two ghost-cell boundary faces.
#   cell residual = (4 + 2·2)/h² × u = 0... let us derive from the assembled A.
#
# A (computed by unit-basis assembly) for shape=(2,2), h=1/2:
#   A_{ii} = 24 for all i (each cell touches 2 boundary faces out of 4)
#   A_{ij} = -4 for adjacent cells, 0 otherwise
#   A @ [1,1,1,1]^T = [24-4-4, 24-4-4, 24-4-4, 24-4-4] = [16,16,16,16]
# ---------------------------------------------------------------------------

_rhs_2d = LazyMeshFunction(_mesh_2d, lambda _idx: sympy.Integer(16))
_exact_2d = LazyMeshFunction(_mesh_2d, lambda _idx: 1.0)

# ---------------------------------------------------------------------------
# Solver cases: (solver, disc, rhs, u_exact, atol)
# ---------------------------------------------------------------------------

_SOLVER_CASES = [
    (DenseJacobiSolver(tol=1e-10), _disc_1d, _rhs_1d, _exact_1d, 1e-6),
    (DenseJacobiSolver(tol=1e-10), _disc_2d, _rhs_2d, _exact_2d, 1e-6),
]

# ---------------------------------------------------------------------------
# Spectral cases: discretizations whose Jacobi matrix is verified ρ < 1
# ---------------------------------------------------------------------------

_SPECTRAL_CASES = [
    _disc_1d,
    _disc_2d,
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _l2h_norm(u: LazyMeshFunction, v: LazyMeshFunction, mesh: CartesianMesh) -> float:
    """Discrete L²_h norm of (u − v): (Σᵢ |Ωᵢ| (uᵢ − vᵢ)²)^{1/2}."""
    shape = mesh.shape
    ndim = len(shape)
    n = math.prod(shape)
    vol = float(mesh.cell_volume)

    def to_multi(flat: int) -> tuple[int, ...]:
        idx = []
        k = flat
        for axis in range(ndim):
            idx.append(k % shape[axis])
            k //= shape[axis]
        return tuple(idx)

    sq: float = sum(  # type: ignore[assignment]
        vol * (float(u(to_multi(i))) - float(v(to_multi(i)))) ** 2  # type: ignore[arg-type]
        for i in range(n)
    )
    return sq**0.5


@pytest.mark.parametrize(
    "solver,disc,rhs,expected_u,atol",
    _SOLVER_CASES,
    ids=[
        f"{type(s).__name__}({type(d.mesh).__name__}{d.mesh.shape})"
        for s, d, *_ in _SOLVER_CASES
    ],
)
def test_linear_solver_solves(
    solver: DenseJacobiSolver,
    disc: FVMDiscretization,
    rhs: LazyMeshFunction,
    expected_u: LazyMeshFunction,
    atol: float,
) -> None:
    u = solver.solve(disc, rhs)
    err = _l2h_norm(u, expected_u, disc.mesh)  # type: ignore[arg-type]
    assert err < atol, f"‖u − u_exact‖_{{L²_h}} = {err:.3e} >= atol = {atol:.3e}"


@pytest.mark.parametrize(
    "disc",
    _SPECTRAL_CASES,
    ids=[f"{type(d.mesh).__name__}{d.mesh.shape}" for d in _SPECTRAL_CASES],
)
def test_jacobi_spectral_radius_less_than_one(disc: FVMDiscretization) -> None:
    """Verify ρ(M_J) < 1 symbolically, guaranteeing Jacobi convergence.

    M_J = I − D⁻¹A where D = diag(A).  For the ghost-cell Dirichlet Poisson
    operator, SPD (C6) and Taussky's theorem imply ρ < 1; this test checks
    the claim directly on the assembled matrix.
    """
    A = disc.assemble_matrix()
    n = A.shape[0]
    d_inv = sympy.diag(*[sympy.Integer(1) / A[i, i] for i in range(n)])
    m_j = sympy.eye(n) - d_inv * A
    for ev in m_j.eigenvals():
        val = complex(ev.evalf())
        assert abs(val) < 1, (
            f"Jacobi eigenvalue {ev} has |λ| = {abs(val):.6f} >= 1; "
            f"Jacobi iteration diverges for {type(disc.mesh).__name__}{disc.mesh.shape}"
        )
