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

import numpy as np
import pytest
import sympy

from cosmic_foundry.computation.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.dense_lu_solver import DenseLUSolver
from cosmic_foundry.geometry.advection_diffusion_flux import AdvectionDiffusionFlux
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
from cosmic_foundry.theory.continuous.periodic_bc import PeriodicBC
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

    Builds the discretization from flux, mesh, and bc_type, then verifies:
      1. Final residual ‖f − Au‖_{L²_h} < tol.
    No monotonicity or iteration-count checks: a direct solver produces the
    exact solution (up to floating-point rounding) in a single pass.

    For PeriodicBC the RHS is a zero-mean sinusoid so the system is consistent
    (in the column space of the circulant advection matrix).
    """

    def __init__(
        self,
        solver: DenseLUSolver,
        flux: Any,
        mesh: CartesianMesh,
        bc_type: type = DirichletBC,
    ) -> None:
        self._solver = solver
        self._flux = flux
        self._mesh = mesh
        self._bc_type = bc_type

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        suffix = "/periodic" if self._bc_type is PeriodicBC else ""
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._flux).__name__}(order={self._flux.order})"
            f"/N={n}{suffix}"
        )

    def check(self) -> None:
        manifold = self._flux.continuous_operator.manifold
        bc = self._bc_type(manifold)
        disc = FVMDiscretization(self._mesh, self._flux, bc)
        if self._bc_type is PeriodicBC:
            h = float(self._mesh.cell_volume)
            orig = float(self._mesh.coordinate((0,))[0]) - 0.5 * h
            rhs = LazyMeshFunction(
                self._mesh,
                lambda idx, _h=h, _o=orig: (
                    math.sin(2 * math.pi * (_o + (idx[0] + 1) * _h))
                    - math.sin(2 * math.pi * (_o + idx[0] * _h))
                )
                / _h,
            )
        else:
            rhs = LazyMeshFunction(self._mesh, lambda idx: 1.0)
        self._solver.solve(disc, rhs)
        r = self._solver.residuals
        assert r[-1] < self._solver._tol, f"Direct solve residual {r[-1]:.3e} >= tol"


class _ConvergenceRateClaim(_Claim):
    """Claim: ‖φ_h − Rₕ φ_exact‖_{L²_h} converges at O(h^p) over the mesh sequence.

    The manufactured solution φ is selected automatically from candidates
    sin(nπx) for n=1..k_max, where k_max = N_min // p gives at least 2p
    cells per wavelength on the coarsest mesh (sufficient for the asymptotic
    regime).  A candidate is admitted only when A·R_h(sin(nπx)) matches
    R_h(∇·F(sin(nπx))) to within 10% on the coarsest mesh; modes inconsistent
    with the BC's ghost-cell convention (e.g. odd-n modes under PeriodicBC)
    produce O(1) relative error and are excluded automatically.  The source
    ρ = ∇·F(φ) is derived symbolically from flux.continuous_operator.

    Before measuring the L²_h error the assembled stiffness matrix is
    decomposed via SVD; any null-space components of u_h are projected out,
    isolating truncation error from the arbitrary null-space component a
    direct solver may introduce for singular systems (e.g. advection under
    PeriodicBC).
    """

    def __init__(
        self,
        solver: Any,
        flux: Any,
        meshes: list[CartesianMesh],
        bc_type: type = DirichletBC,
    ) -> None:
        self._solver = solver
        self._flux = flux
        self._meshes = meshes
        self._bc_type = bc_type

    @property
    def description(self) -> str:
        suffix = "/periodic" if self._bc_type is PeriodicBC else ""
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._flux).__name__}(order={self._flux.order})/"
            f"convergence_rate{suffix}"
        )

    def check(self) -> None:
        manifold = self._flux.continuous_operator.manifold
        _x = manifold.atlas[0].symbols[0]
        p = self._flux.order
        bc = self._bc_type(manifold)

        # Auto-select admissible manufactured-solution modes.
        # k_max = N_min // p ensures >= 2p cells/wavelength on the coarsest mesh.
        # Each candidate sin(nπx) is tested for BC consistency: if
        # ||A·R_h(φ_n) - R_h(ρ_n)|| / ||R_h(ρ_n)|| >= 0.1 the mode violates
        # the ghost-cell convention (O(1) boundary error) and is dropped.
        coarse = min(self._meshes, key=lambda m: m.shape[0])
        n_c = coarse.shape[0]
        vol_c = float(coarse.cell_volume)
        orig_c = float(coarse.coordinate((0,))[0]) - 0.5 * vol_c
        a_sym_c = FVMDiscretization(coarse, self._flux, bc).assemble_matrix()
        a_c = np.array([[float(a_sym_c[i, j]) for j in range(n_c)] for i in range(n_c)])
        k_max = max(1, n_c // p)
        phi_terms: list[sympy.Expr] = []
        for n in range(1, k_max + 1):
            phi_n = sympy.sin(n * sympy.pi * _x)
            one_form_n = self._flux.continuous_operator(
                ZeroForm(manifold, phi_n, (_x,))
            )
            rho_n = sum(
                sympy.diff(one_form_n.component(i), one_form_n.symbols[i])
                for i in range(len(one_form_n.symbols))
            )
            F_pn = sympy.lambdify(_x, sympy.integrate(phi_n, _x), "math")
            F_rn = sympy.lambdify(_x, sympy.integrate(rho_n, _x), "math")
            v_n = np.array(
                [
                    (F_pn(orig_c + (i + 1) * vol_c) - F_pn(orig_c + i * vol_c)) / vol_c
                    for i in range(n_c)
                ]
            )
            r_n = np.array(
                [
                    (F_rn(orig_c + (i + 1) * vol_c) - F_rn(orig_c + i * vol_c)) / vol_c
                    for i in range(n_c)
                ]
            )
            rel_err = float(np.linalg.norm(a_c @ v_n - r_n)) / (
                float(np.linalg.norm(r_n)) + 1e-30
            )
            if rel_err < 0.1:
                phi_terms.append(phi_n)
        assert phi_terms, "No admissible manufactured-solution modes found"
        phi_expr = sympy.Add(*phi_terms)
        phi = ZeroForm(manifold, phi_expr, (_x,))

        # Derive ρ = ∇·F(φ) symbolically from the flux's continuous operator.
        one_form = self._flux.continuous_operator(phi)
        rho_expr = sum(
            sympy.diff(one_form.component(i), one_form.symbols[i])
            for i in range(len(one_form.symbols))
        )

        # Lambdify antiderivatives for O(1) float cell-average evaluation:
        # cell_avg(f, i) = (F(x_{i+1}) − F(x_i)) / h  where F' = f.
        F_phi = sympy.lambdify(_x, sympy.integrate(phi_expr, _x), "math")
        F_rho = sympy.lambdify(_x, sympy.integrate(rho_expr, _x), "math")

        errors: list[float] = []
        for mesh in self._meshes:
            vol = float(mesh.cell_volume)
            orig = float(mesh.coordinate((0,))[0]) - 0.5 * vol
            n_cells = mesh.shape[0]

            def _phi_avg(i: int, _v: float = vol, _o: float = orig) -> float:
                return (F_phi(_o + (i + 1) * _v) - F_phi(_o + i * _v)) / _v

            def _rho_avg(i: int, _v: float = vol, _o: float = orig) -> float:
                return (F_rho(_o + (i + 1) * _v) - F_rho(_o + i * _v)) / _v

            disc = FVMDiscretization(mesh, self._flux, bc)

            # Project u_h onto the orthogonal complement of the assembled
            # matrix's null space before measuring truncation error.  The SVD
            # threshold is relative to the largest singular value; for full-rank
            # systems no null vectors are found and the projection is a no-op.
            a_sym = disc.assemble_matrix()
            a_np = np.array(
                [[float(a_sym[i, j]) for j in range(n_cells)] for i in range(n_cells)]
            )
            _, s_vals, vt = np.linalg.svd(a_np)
            null_tol = s_vals[0] * n_cells * float(np.finfo(float).eps) ** 0.5
            null_vecs = vt[s_vals < null_tol]

            rhs = LazyMeshFunction(mesh, lambda idx, _r=_rho_avg: _r(idx[0]))
            u_h = self._solver.solve(disc, rhs)

            u_arr = np.array([float(u_h((i,))) for i in range(n_cells)])
            for v in null_vecs:
                u_arr -= float(np.dot(u_arr, v)) * v
            phi_arr = np.array([_phi_avg(i) for i in range(n_cells)])
            err_sq = float(vol * np.dot(u_arr - phi_arr, u_arr - phi_arr))
            errors.append(math.sqrt(err_sq))

        hs = [float(m.cell_volume) for m in self._meshes]
        log_h = [math.log(hv) for hv in hs]
        log_e = [math.log(ev) for ev in errors]
        n_pts = len(log_h)
        sx = sum(log_h)
        sy = sum(log_e)
        sxy = sum(lh * le for lh, le in zip(log_h, log_e, strict=False))
        sxx = sum(lh * lh for lh in log_h)
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

_DIFFUSIVE_FLUXES = [
    DiffusiveFlux(DiffusiveFlux.min_order, _manifold),
    DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold),
]
_ADVECTIVE_FLUXES = [
    AdvectiveFlux(AdvectiveFlux.min_order, _manifold),
    AdvectiveFlux(AdvectiveFlux.min_order + AdvectiveFlux.order_step, _manifold),
]
_ADVECTION_DIFFUSION_FLUXES = [
    AdvectionDiffusionFlux(AdvectionDiffusionFlux.min_order, _manifold),
    AdvectionDiffusionFlux(
        AdvectionDiffusionFlux.min_order + AdvectionDiffusionFlux.order_step, _manifold
    ),
]
_FLUXES = [*_DIFFUSIVE_FLUXES, *_ADVECTIVE_FLUXES, *_ADVECTION_DIFFUSION_FLUXES]
# DiffusiveFlux assembles an SPD matrix (DirichletBC); compatible with all solvers.
# AdvectiveFlux assembles a rank-(N-1) circulant matrix under PeriodicBC; compatible
# with DenseLUSolver only (zero-mean null-space convention handles the singularity).
# AdvectionDiffusionFlux assembles A_adv + κ·A_diff; non-singular under DirichletBC
# for any κ > 0, compatible with all solvers at unit Péclet number (κ=1, h≈1/N).
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
    # Diffusive (SPD, DirichletBC): all solvers
    *[_SolverClaim(s, f, _mesh_n8) for s in _SOLVERS for f in _DIFFUSIVE_FLUXES],
    *[
        _DirectSolverClaim(s, f, _mesh_n8)
        for s in _DIRECT_SOLVERS
        for f in _DIFFUSIVE_FLUXES
    ],
    *[
        _ConvergenceRateClaim(s, f, _CONVERGENCE_MESHES)
        for s in [*_SOLVERS, *_DIRECT_SOLVERS]
        for f in _DIFFUSIVE_FLUXES
    ],
    # Advective (rank-(N-1) circulant, PeriodicBC): direct solver only
    *[
        _DirectSolverClaim(s, f, _mesh_n8, PeriodicBC)
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES
    ],
    *[
        _ConvergenceRateClaim(s, f, _CONVERGENCE_MESHES, PeriodicBC)
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES
    ],
    # Advection-diffusion (non-singular under DirichletBC for κ>0): all solvers
    *[
        _SolverClaim(s, f, _mesh_n8)
        for s in _SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES
    ],
    *[
        _DirectSolverClaim(s, f, _mesh_n8)
        for s in _DIRECT_SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES
    ],
    *[
        _ConvergenceRateClaim(s, f, _CONVERGENCE_MESHES)
        for s in [*_SOLVERS, *_DIRECT_SOLVERS]
        for f in _ADVECTION_DIFFUSION_FLUXES
    ],
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_convergence(claim: _Claim) -> None:
    claim.check()
