"""Convergence verification for all concrete DiscreteOperator subclasses and solvers.

Each convergence claim is a _Claim subclass that encodes both what is being
verified and how to verify it.  Adding a new claim requires only appending to
_CLAIMS; the single parametric test covers all entries.

  _OrderClaim(instance)               — instance achieves O(h^p) at declared p
  _SolverClaim(solver, flux, mesh)    — iterative solver reaches tol with
                                        monotonically decreasing residuals and
                                        analytically bounded iteration count
  _DirectSolverClaim(solver, flux, mesh)
                                      — direct solver residual < tol after one
                                        factorization pass
  _ConvergenceRateClaim(solver, flux, meshes)
                                      — L²_h error converges at >= O(h^{p-0.1})
                                        over a mesh refinement sequence using a
                                        manufactured solution with exact cell
                                        averages for source and reference field

_FLUXES contains all NumericalFlux instances; adding a new flux automatically
generates _OrderClaim entries for it.  _ELLIPTIC_FLUXES is the SPD subset
that also gets solver/convergence claims — fluxes whose assembled stiffness
matrix is not SPD (e.g. AdvectiveFlux) belong only in _FLUXES.  _SOLVERS
(iterative), _DIRECT_SOLVERS (direct), and _CONVERGENCE_MESHES are registries
for the elliptic claims.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.dense_lu_solver import DenseLUSolver
from cosmic_foundry.geometry.advective_flux import AdvectiveFlux
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianRestrictionOperator,
)
from cosmic_foundry.geometry.diffusive_flux import DiffusiveFlux
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.geometry.fvm_discretization import FVMDiscretization
from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    ZeroForm,
)
from cosmic_foundry.theory.continuous.dirichlet_bc import DirichletBC
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction


class _Claim(ABC):
    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self) -> None: ...


class _OrderClaim(_Claim):
    """Claim: discrete operator achieves O(h^p) convergence at order p.

    Verifies that the error polynomial has zeros at h⁰…h^{p-1} and a
    nonzero h^p leading term, using a manufactured polynomial solution
    on a 1-D symbolic mesh.  The exact value is computed as Rₕ(L φ)
    via CartesianRestrictionOperator — no per-instance oracle.
    """

    def __init__(self, instance: Any) -> None:
        self._instance = instance

    @property
    def description(self) -> str:
        return f"{type(self._instance).__name__}(order={self._instance.order})"

    def check(self) -> None:
        instance = self._instance
        h = sympy.Symbol("h", positive=True)
        order = instance.order
        n = order // 2

        space = EuclideanManifold(1)
        x = space.atlas[0].symbols[0]
        mesh = CartesianMesh(
            origin=(sympy.Integer(0),),
            spacing=(h,),
            shape=(2 * n + 2,),
        )
        ndim = len(mesh._shape)

        coeffs = sympy.symbols(f"a:{order + 4}")
        phi_expr: sympy.Expr = sum(c * x**k for k, c in enumerate(coeffs))
        phi = ZeroForm(space, phi_expr, (x,))

        U = CartesianRestrictionOperator(mesh, degree=ndim)(phi)
        numerical_mf = instance(U)

        cont_result = instance.continuous_operator(phi)
        assert isinstance(cont_result, DifferentialForm)
        restriction_degree = ndim - cont_result.degree
        exact_mf = CartesianRestrictionOperator(mesh, degree=restriction_degree)(
            cont_result
        )

        test_idx: Any = (0, (n,)) if restriction_degree < ndim else (n,)
        error = sympy.expand(
            sympy.simplify(numerical_mf(test_idx) - exact_mf(test_idx))
        )
        poly = sympy.Poly(error, h)
        for k in range(order):
            assert poly.nth(k) == 0, (
                f"Unexpected O(h^{k}) term in {type(instance).__name__}"
                f"(order={order}): {poly.nth(k)}"
            )
        assert poly.nth(order) != 0, (
            f"Missing O(h^{order}) leading term in "
            f"{type(instance).__name__}(order={order})"
        )


class _SolverClaim(_Claim):
    """Claim: solver converges on FVMDiscretization(mesh, flux, DirichletBC).

    Builds the discretization from flux and mesh, then verifies:
      1. Final residual < tol.
      2. ‖f − Au^k‖_{L²_h} decreases at every step.
      3. Asymptotic convergence rate (geometric mean of last 20 ratios) < 1.
      4. Iteration count ≤ the upper bound implied by that rate and the
         initial residual.
    """

    def __init__(
        self,
        solver: DenseJacobiSolver,
        flux: Any,
        mesh: CartesianMesh,
    ) -> None:
        self._solver = solver
        self._flux = flux
        self._mesh = mesh

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._flux).__name__}(order={self._flux.order})/N={n}"
        )

    def check(self) -> None:
        manifold = self._flux.continuous_operator.manifold
        disc = FVMDiscretization(self._mesh, self._flux, DirichletBC(manifold))
        rhs = LazyMeshFunction(self._mesh, lambda idx: 1.0)
        self._solver.solve(disc, rhs)
        r = self._solver.residuals

        assert r[-1] < self._solver._tol, f"Did not converge: final residual {r[-1]}"

        for k in range(1, len(r)):
            assert (
                r[k] <= r[k - 1]
            ), f"Non-monotone residual at step {k}: {r[k]:.3e} > {r[k - 1]:.3e}"

        tail_len = min(20, len(r) - 1)
        rho = (r[-1] / r[-1 - tail_len]) ** (1.0 / tail_len)
        assert rho < 1.0, f"Asymptotic rate {rho:.6f} >= 1; solver not contracting"

        k_bound = math.ceil(math.log(self._solver._tol / r[0]) / math.log(rho))
        assert (
            len(r) <= k_bound
        ), f"Iteration count {len(r)} exceeds spectral-radius bound {k_bound}"


class _DirectSolverClaim(_Claim):
    """Claim: direct solver residual < tol after one factorization pass.

    Builds the discretization from flux and mesh, then verifies:
      1. Final residual ‖f − Au‖_{L²_h} < tol.
    No monotonicity or iteration-count checks: a direct solver produces the
    exact solution (up to floating-point rounding) in a single pass.
    """

    def __init__(
        self,
        solver: DenseLUSolver,
        flux: Any,
        mesh: CartesianMesh,
    ) -> None:
        self._solver = solver
        self._flux = flux
        self._mesh = mesh

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._flux).__name__}(order={self._flux.order})/N={n}"
        )

    def check(self) -> None:
        manifold = self._flux.continuous_operator.manifold
        disc = FVMDiscretization(self._mesh, self._flux, DirichletBC(manifold))
        rhs = LazyMeshFunction(self._mesh, lambda idx: 1.0)
        self._solver.solve(disc, rhs)
        r = self._solver.residuals
        assert r[-1] < self._solver._tol, f"Direct solve residual {r[-1]:.3e} >= tol"


class _ConvergenceRateClaim(_Claim):
    """Claim: ‖φ_h − Rₕ φ_exact‖_{L²_h} converges at O(h^p) over the mesh sequence."""

    def __init__(
        self,
        solver: Any,
        flux: Any,
        meshes: list[CartesianMesh],
    ) -> None:
        self._solver = solver
        self._flux = flux
        self._meshes = meshes

    @property
    def description(self) -> str:
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._flux).__name__}(order={self._flux.order})/"
            "convergence_rate"
        )

    def check(self) -> None:
        manifold = self._flux.continuous_operator.manifold
        errors: list[float] = []
        for mesh in self._meshes:
            h = float(mesh.cell_volume)  # 1-D: cell_volume == spacing
            orig = float(mesh.coordinate((0,))[0]) - 0.5 * h

            def _phi_avg(i: int, _h: float = h, _o: float = orig) -> float:
                xl, xr = _o + i * _h, _o + (i + 1) * _h
                return (math.cos(math.pi * xl) - math.cos(math.pi * xr)) / (
                    math.pi * _h
                ) + (math.cos(3 * math.pi * xl) - math.cos(3 * math.pi * xr)) / (
                    3 * math.pi * _h
                )

            def _rho_avg(i: int, _h: float = h, _o: float = orig) -> float:
                xl, xr = _o + i * _h, _o + (i + 1) * _h
                return (
                    math.pi * (math.cos(math.pi * xl) - math.cos(math.pi * xr)) / _h
                    + 3
                    * math.pi
                    * (math.cos(3 * math.pi * xl) - math.cos(3 * math.pi * xr))
                    / _h
                )

            disc = FVMDiscretization(mesh, self._flux, DirichletBC(manifold))
            rhs = LazyMeshFunction(mesh, lambda idx, _r=_rho_avg: _r(idx[0]))
            u_h = self._solver.solve(disc, rhs)
            vol = float(mesh.cell_volume)
            # All cells included: for homogeneous Dirichlet BC, odd-reflection ghost
            # values equal the exact odd-extension cell averages of any solution
            # satisfying φ(0)=φ(1)=0, so boundary cells converge at full order p.
            err_sq: float = sum(
                vol * (float(u_h((i,))) - _phi_avg(i)) ** 2
                for i in range(mesh.shape[0])
            )
            errors.append(math.sqrt(err_sq))

        hs = [float(m.cell_volume) for m in self._meshes]
        log_h = [math.log(hv) for hv in hs]
        log_e = [math.log(ev) for ev in errors]
        n_pts = len(log_h)
        sx = sum(log_h)
        sy = sum(log_e)
        sxy = sum(x * y for x, y in zip(log_h, log_e, strict=False))
        sxx = sum(x * x for x in log_h)
        slope = (n_pts * sxy - sx * sy) / (n_pts * sxx - sx**2)
        assert slope >= self._flux.order - 0.1, (
            f"Convergence rate {slope:.3f} < expected "
            f"{self._flux.order - 0.1:.1f} for "
            f"{type(self._flux).__name__}(order={self._flux.order})"
        )


_manifold = EuclideanManifold(1)
_dummy_mesh = CartesianMesh(
    origin=(sympy.Integer(0),), spacing=(sympy.Integer(1),), shape=(4,)
)
_mesh_n8 = CartesianMesh(
    origin=(sympy.Rational(0),),
    spacing=(sympy.Rational(1, 8),),
    shape=(8,),
)

_FLUXES = [
    DiffusiveFlux(DiffusiveFlux.min_order, _manifold),
    DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold),
    AdvectiveFlux(AdvectiveFlux.min_order, _manifold),
    AdvectiveFlux(AdvectiveFlux.min_order + AdvectiveFlux.order_step, _manifold),
]
# SPD subset: only fluxes whose assembled stiffness matrix is SPD are compatible
# with DenseJacobiSolver; AdvectiveFlux produces a skew-symmetric matrix.
_ELLIPTIC_FLUXES = [f for f in _FLUXES if isinstance(f, DiffusiveFlux)]
_SOLVERS = [DenseJacobiSolver(tol=1e-8)]
_DIRECT_SOLVERS = [DenseLUSolver(tol=1e-10)]
_CONVERGENCE_MESHES = [
    CartesianMesh(
        origin=(sympy.Rational(0),),
        spacing=(sympy.Rational(1, n),),
        shape=(n,),
    )
    for n in [16, 24, 32, 48, 64]
]

_CLAIMS: list[_Claim] = [
    *[_OrderClaim(f) for f in _FLUXES],
    *[_OrderClaim(FVMDiscretization(_dummy_mesh, f)()) for f in _FLUXES],
    *[_SolverClaim(s, f, _mesh_n8) for s in _SOLVERS for f in _ELLIPTIC_FLUXES],
    *[
        _DirectSolverClaim(s, f, _mesh_n8)
        for s in _DIRECT_SOLVERS
        for f in _ELLIPTIC_FLUXES
    ],
    *[
        _ConvergenceRateClaim(s, f, _CONVERGENCE_MESHES)
        for s in [*_SOLVERS, *_DIRECT_SOLVERS]
        for f in _ELLIPTIC_FLUXES
    ],
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_convergence(claim: _Claim) -> None:
    claim.check()
