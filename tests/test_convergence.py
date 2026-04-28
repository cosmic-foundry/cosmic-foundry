"""Convergence verification for all concrete DiscreteOperator subclasses and solvers.

Each convergence claim is a CalibratedClaim subclass that encodes both what is
being verified and how to verify it.  Adding a new claim requires only appending
to _CLAIMS; the single parametric test covers all entries.

  _OrderClaim(instance)               — instance achieves O(h^p) at declared p
  _SolverClaim(solver, flux, mesh)    — iterative solver residual < tol
  _DirectSolverClaim(solver, flux, mesh)
                                      — direct solver residual < tol after one
                                        factorization pass
  _ConvergenceRateClaim(solver, flux)
                                      — L²_h error converges at >= O(h^{p-0.1})
                                        over a mesh refinement sequence using a
                                        manufactured solution with exact cell
                                        averages for source and reference field

_FLUXES contains all NumericalFlux instances; adding a new flux automatically
generates _OrderClaim entries for it.  _SOLVERS (iterative, non-SPD-restricted)
and _SPD_SOLVERS (iterative, SPD-only) and _DIRECT_SOLVERS (direct) are solver
registries.  Diffusive claims use all three; advection-diffusion claims exclude
_SPD_SOLVERS because that matrix is non-symmetric.  The convergence mesh
sequence is computed adaptively at runtime from the machine's FMA rate; see
_convergence_n_max.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation import tensor
from cosmic_foundry.computation.decompositions.svd_factorization import SVDFactorization
from cosmic_foundry.computation.solvers.dense_cg_solver import DenseCGSolver
from cosmic_foundry.computation.solvers.dense_gauss_seidel_solver import (
    DenseGaussSeidelSolver,
)
from cosmic_foundry.computation.solvers.dense_gmres_solver import DenseGMRESSolver
from cosmic_foundry.computation.solvers.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.solvers.dense_lu_solver import DenseLUSolver
from cosmic_foundry.computation.solvers.dense_svd_solver import DenseSVDSolver
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianFaceRestriction,
    CartesianVolumeRestriction,
)
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.physics.advection_diffusion_flux import AdvectionDiffusionFlux
from cosmic_foundry.physics.advective_flux import AdvectiveFlux
from cosmic_foundry.physics.diffusive_flux import DiffusiveFlux
from cosmic_foundry.physics.operator import Operator
from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ZeroForm,
)
from cosmic_foundry.theory.continuous.differential_operator import DivergenceComposition
from cosmic_foundry.theory.continuous.diffusion_operator import DiffusionOperator
from cosmic_foundry.theory.discrete import (
    DirichletGhostCells,
    FDDiscretization,
    FVMDiscretization,
    PeriodicGhostCells,
)
from cosmic_foundry.theory.discrete.discrete_field import _CallableDiscreteField
from tests.calibration import _MESH_FRACTIONS, _NP_BACKEND, _convergence_n_max
from tests.claims import CalibratedClaim

# ---------------------------------------------------------------------------
# Claim classes
# ---------------------------------------------------------------------------


class _OrderClaim(CalibratedClaim[float]):
    """Claim: discrete operator achieves O(h^p) convergence at order p.

    Verifies that the error polynomial has zeros at h⁰…h^{p-1} and a
    nonzero h^p leading term, using a manufactured polynomial solution
    on a 1-D symbolic mesh.

    point_dof=False (default): input field is cell-averaged; exact reference
    is computed via CartesianRestrictionOperator (FVM convention).
    point_dof=True: input field is point-evaluated at cell centers; exact
    reference is the continuous operator evaluated at the same points (FD
    convention).
    """

    def __init__(self, instance: Any, *, point_dof: bool = False) -> None:
        self._instance = instance
        self._point_dof = point_dof

    @property
    def description(self) -> str:
        return f"{type(self._instance).__name__}(order={self._instance.order})"

    def check(self, fma_rate: float) -> None:
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

        coeffs = sympy.symbols(f"a:{order + 4}")
        phi_expr: sympy.Expr = sum(c * x**k for k, c in enumerate(coeffs))
        phi = ZeroForm(space, phi_expr, (x,))

        def _x_at(idx: tuple[int, ...]) -> sympy.Expr:
            return (
                mesh._origin[0]
                + (sympy.Integer(idx[0]) + sympy.Rational(1, 2)) * mesh._spacing[0]
            )

        if self._point_dof:
            U = _CallableDiscreteField(mesh, lambda idx: phi_expr.subs(x, _x_at(idx)))
            numerical_mf = instance(U)
            cont_result = instance.continuous_operator(phi)
            exact_val = cont_result.expr.subs(x, _x_at((n,)))
            error = sympy.expand(sympy.simplify(numerical_mf((n,)) - exact_val))
        else:
            vol = mesh.cell_volume
            U_totals = CartesianVolumeRestriction(mesh)(phi)
            U_avg = _CallableDiscreteField(mesh, lambda idx, _U=U_totals: _U(idx) / vol)
            numerical_mf = instance(U_avg)
            cont_result = instance.continuous_operator(phi)
            assert isinstance(cont_result, DifferentialForm)
            if isinstance(cont_result, ZeroForm):
                exact_mf = CartesianVolumeRestriction(mesh)(cont_result)
                test_idx: Any = (n,)
                error = sympy.expand(
                    sympy.simplify(numerical_mf(test_idx) - exact_mf(test_idx) / vol)
                )
            else:
                assert isinstance(cont_result, OneForm)
                exact_mf = CartesianFaceRestriction(mesh)(cont_result)
                test_idx = (0, (n,))
                error = sympy.expand(
                    sympy.simplify(numerical_mf(test_idx) - exact_mf(test_idx))
                )

        expected_leading = order
        poly = sympy.Poly(error, h)
        for k in range(expected_leading):
            assert poly.nth(k) == 0, (
                f"Unexpected O(h^{k}) term in {type(instance).__name__}"
                f"(order={order}): {poly.nth(k)}"
            )
        assert poly.nth(expected_leading) != 0, (
            f"Missing O(h^{expected_leading}) leading term in "
            f"{type(instance).__name__}(order={order})"
        )


class _SolverClaim(CalibratedClaim[float]):
    """Claim: solver residual < tol after solve on the given Discretization.

    Verifies that ‖b − Au‖₂ < tol after solve returns.  disc is pre-built
    with its BC; Operator binds it to mesh at check time.
    """

    def __init__(self, solver: Any, disc: Any, mesh: CartesianMesh) -> None:
        self._solver = solver
        self._disc = disc
        self._mesh = mesh

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._disc).__name__}(order={self._disc.order})/N={n}"
        )

    def check(self, fma_rate: float) -> None:
        n = math.prod(self._mesh.shape)
        op = Operator(self._disc, self._mesh)
        b = Tensor([1.0] * n, backend=_NP_BACKEND)
        u = self._solver.solve(op, b)
        residual = tensor.norm(b - op.apply(u)).get()
        tol = getattr(self._solver, "_tol", 1e-10)
        assert residual < tol, f"Did not converge: residual {residual:.3e}"


class _DirectSolverClaim(CalibratedClaim[float]):
    """Claim: direct solver residual < tol after one factorization pass.

    disc is pre-built with its BC; Operator binds it to mesh at check time.
    For PeriodicGhostCells the RHS is a zero-mean sinusoid so the system is
    consistent (in the column space of the circulant advection matrix).
    """

    def __init__(self, solver: Any, disc: Any, mesh: CartesianMesh) -> None:
        self._solver = solver
        self._disc = disc
        self._mesh = mesh

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        periodic = isinstance(self._disc.boundary_condition, PeriodicGhostCells)
        suffix = "/periodic" if periodic else ""
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._disc).__name__}(order={self._disc.order})"
            f"/N={n}{suffix}"
        )

    def check(self, fma_rate: float) -> None:
        n = math.prod(self._mesh.shape)
        op = Operator(self._disc, self._mesh)
        if isinstance(self._disc.boundary_condition, PeriodicGhostCells):
            h = float(self._mesh.cell_volume)
            orig = float(self._mesh.coordinate((0,))[0]) - 0.5 * h
            b = Tensor(
                [
                    (
                        math.sin(2 * math.pi * (orig + (i + 1) * h))
                        - math.sin(2 * math.pi * (orig + i * h))
                    )
                    / h
                    for i in range(n)
                ]
            )
        else:
            b = Tensor([1.0] * n)
        u = self._solver.solve(op, b)
        residual = tensor.norm(b - op.apply(u)).get()
        assert residual < 1e-10, f"Direct solve residual {residual:.3e} >= 1e-10"


class _ConvergenceRateClaim(CalibratedClaim[float]):
    """Claim: ‖φ_h − φ_exact‖_{L²_h} converges at O(h^p) over the mesh sequence.

    disc is pre-built with its BC; Operator binds it to each mesh at check time.
    disc.continuous_operator (a ZeroForm → ZeroForm operator) is used to derive
    the manufactured-solution source ρ = L(φ) and to auto-select admissible
    sinusoidal modes.  disc.order drives the k_max and slope assertions.

    point_dof=False (default): cell-average DOF convention; ρ and φ are evaluated
    as cell averages via antiderivative lambdification (FVM convention).
    point_dof=True: point-value DOF convention; ρ and φ are evaluated at cell
    centers (FD convention).

    Admissible modes are those for which A_h·φ_n ≈ ρ_n to within 10% on the
    coarsest mesh.  For PeriodicGhostCells only even-n modes are candidates:
    sin(nπx) is 1-periodic iff n is even.

    Before measuring L²_h error the stiffness matrix is SVD-decomposed; any
    null-space components of u_h are projected out, isolating truncation error
    from arbitrary null-space contributions (e.g. advection under PeriodicBC).
    """

    def __init__(self, solver: Any, disc: Any, *, point_dof: bool = False) -> None:
        self._solver = solver
        self._disc = disc
        self._point_dof = point_dof

    @property
    def description(self) -> str:
        periodic = isinstance(self._disc.boundary_condition, PeriodicGhostCells)
        suffix = "/periodic" if periodic else ""
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._disc).__name__}(order={self._disc.order})/"
            f"convergence_rate{suffix}"
        )

    def check(self, fma_rate: float) -> None:
        n_max = _convergence_n_max(fma_rate, _N_CONVERGENCE_CLAIMS, self._solver)
        meshes = [
            CartesianMesh(
                origin=(sympy.Rational(0),),
                spacing=(sympy.Rational(1, int(n_max * f)),),
                shape=(int(n_max * f),),
            )
            for f in _MESH_FRACTIONS
        ]

        cont_op = self._disc.continuous_operator
        manifold = cont_op.manifold
        _x = manifold.atlas[0].symbols[0]
        p = self._disc.order
        periodic = isinstance(self._disc.boundary_condition, PeriodicGhostCells)

        # Pre-build all operators and SVD decompositions for null-space extraction.
        assembled: list[tuple[Any, list[Any], float, float, int]] = []
        for mesh in meshes:
            vol = float(mesh.cell_volume)
            orig = float(mesh.coordinate((0,))[0]) - 0.5 * vol
            n_cells = mesh.shape[0]
            op_m = Operator(self._disc, mesh)
            a_m = _assemble_from_op(op_m, n_cells, _NP_BACKEND)
            decomp = SVDFactorization().factorize(a_m)
            s_vec = decomp.s
            vt = decomp.vt
            null_tol = float(s_vec[0]) * n_cells * sys.float_info.epsilon**0.5
            null_vecs = [vt[j] for j in range(n_cells) if float(s_vec[j]) < null_tol]
            assembled.append((op_m, null_vecs, vol, orig, n_cells))

        # Auto-select admissible manufactured-solution modes.
        # k_max = N_min // p ensures >= 2p cells/wavelength on the coarsest mesh.
        # For PeriodicGhostCells only even-n candidates are tested: sin(nπx) is
        # 1-periodic iff n is even, so odd modes are algebraically incompatible.
        op_c, _, vol_c, orig_c, n_c = assembled[0]
        k_max = max(1, n_c // p)
        step = 2 if periodic else 1
        phi_terms: list[sympy.Expr] = []
        for n in range(step, k_max + 1, step):
            phi_n = sympy.sin(n * sympy.pi * _x)
            rho_n = cont_op(ZeroForm(manifold, phi_n, (_x,))).expr
            if self._point_dof:
                f_pn = sympy.lambdify(_x, phi_n, "math")
                f_rn = sympy.lambdify(_x, rho_n, "math")
                v_n = Tensor(
                    [f_pn(orig_c + (i + 0.5) * vol_c) for i in range(n_c)],
                    backend=_NP_BACKEND,
                )
                r_n = Tensor(
                    [f_rn(orig_c + (i + 0.5) * vol_c) for i in range(n_c)],
                    backend=_NP_BACKEND,
                )
            else:
                F_pn = sympy.lambdify(_x, sympy.integrate(phi_n, _x), "math")
                F_rn = sympy.lambdify(_x, sympy.integrate(rho_n, _x), "math")
                v_n = Tensor(
                    [
                        (F_pn(orig_c + (i + 1) * vol_c) - F_pn(orig_c + i * vol_c))
                        / vol_c
                        for i in range(n_c)
                    ],
                    backend=_NP_BACKEND,
                )
                r_n = Tensor(
                    [
                        (F_rn(orig_c + (i + 1) * vol_c) - F_rn(orig_c + i * vol_c))
                        / vol_c
                        for i in range(n_c)
                    ],
                    backend=_NP_BACKEND,
                )
            rel_err = tensor.norm(op_c.apply(v_n) - r_n).get() / (
                tensor.norm(r_n).get() + 1e-30
            )
            if rel_err < 0.1:
                phi_terms.append(phi_n)
        assert phi_terms, "No admissible manufactured-solution modes found"
        phi_expr = sympy.Add(*phi_terms)
        phi = ZeroForm(manifold, phi_expr, (_x,))

        # Derive ρ = L(φ) symbolically from disc.continuous_operator.
        rho_expr = cont_op(phi).expr

        if self._point_dof:
            f_phi = sympy.lambdify(_x, phi_expr, "math")
            f_rho = sympy.lambdify(_x, rho_expr, "math")
        else:
            F_phi = sympy.lambdify(_x, sympy.integrate(phi_expr, _x), "math")
            F_rho = sympy.lambdify(_x, sympy.integrate(rho_expr, _x), "math")

        errors: list[float] = []
        for op_m, null_vecs, vol, orig, n_cells in assembled:
            if self._point_dof:

                def _phi_pt(i: int, _v: float = vol, _o: float = orig) -> float:
                    return f_phi(_o + (i + 0.5) * _v)

                def _rho_pt(i: int, _v: float = vol, _o: float = orig) -> float:
                    return f_rho(_o + (i + 0.5) * _v)

                b_m = Tensor([_rho_pt(i) for i in range(n_cells)], backend=_NP_BACKEND)
                phi_arr = Tensor(
                    [_phi_pt(i) for i in range(n_cells)], backend=_NP_BACKEND
                )
            else:

                def _phi_avg(i: int, _v: float = vol, _o: float = orig) -> float:
                    return (F_phi(_o + (i + 1) * _v) - F_phi(_o + i * _v)) / _v

                def _rho_avg(i: int, _v: float = vol, _o: float = orig) -> float:
                    return (F_rho(_o + (i + 1) * _v) - F_rho(_o + i * _v)) / _v

                b_m = Tensor([_rho_avg(i) for i in range(n_cells)], backend=_NP_BACKEND)
                phi_arr = Tensor(
                    [_phi_avg(i) for i in range(n_cells)], backend=_NP_BACKEND
                )

            u_arr = self._solver.solve(op_m, b_m)
            for v in null_vecs:
                u_arr = u_arr - float(u_arr @ v) * v
            diff = u_arr - phi_arr
            errors.append(math.sqrt(vol * (diff @ diff)))

        hs = [float(m.cell_volume) for m in meshes]
        log_h = [math.log(hv) for hv in hs]
        log_e = [math.log(ev) for ev in errors]
        n_pts = len(log_h)
        sx = sum(log_h)
        sy = sum(log_e)
        sxy = sum(lh * le for lh, le in zip(log_h, log_e, strict=False))
        sxx = sum(lh * lh for lh in log_h)
        syy = sum(le * le for le in log_e)
        denom_x = n_pts * sxx - sx**2
        slope = (n_pts * sxy - sx * sy) / denom_x
        r2 = (n_pts * sxy - sx * sy) ** 2 / (denom_x * (n_pts * syy - sy**2))
        assert slope >= p - 0.1, (
            f"Convergence rate {slope:.3f} < expected {p - 0.1:.1f} for "
            f"{type(self._disc).__name__}(order={p})"
        )
        assert r2 >= 0.999, (
            f"Convergence not clean power-law: R²={r2:.4f} for "
            f"{type(self._disc).__name__}(order={p})"
        )


def _assemble_from_op(op: Operator, n: int, backend: Any) -> Any:
    """Build the N×N stiffness matrix from op.apply on basis vectors."""
    columns: list[list[float]] = []
    for j in range(n):
        e_j = Tensor.zeros(n, backend=backend)
        e_j = e_j.set(j, Tensor(1.0, backend=backend))
        columns.append(backend.flatten(op.apply(e_j)._value))
    rows = [[columns[j][i] for j in range(n)] for i in range(n)]
    return Tensor(rows, backend=backend)


_manifold = EuclideanManifold(1)
_fd_cont_op = DivergenceComposition(DiffusionOperator(_manifold))
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
# DiffusiveFlux assembles an SPD matrix (DirichletGhostCells);
# compatible with all solvers including CG.
# AdvectiveFlux assembles a rank-(N-1) circulant matrix under PeriodicGhostCells;
# compatible with direct solvers only (zero-mean null-space convention handles the
# singularity).
# AdvectionDiffusionFlux assembles A_adv + κ·A_diff; non-symmetric under
# DirichletGhostCells, so CG is excluded; Jacobi/GS/direct solvers work for κ=1.
_SOLVERS = [
    DenseJacobiSolver(tol=1e-8),
    DenseGaussSeidelSolver(tol=1e-8),
    DenseGMRESSolver(tol=1e-8),
]
_SPD_SOLVERS = [DenseCGSolver(tol=1e-8)]  # SPD (symmetric positive definite) only
_DIRECT_SOLVERS = [DenseLUSolver(), DenseSVDSolver()]


_FD_DISC_2 = FDDiscretization(2, _fd_cont_op, DirichletGhostCells())

_CLAIMS: list[CalibratedClaim[float]] = [
    *[_OrderClaim(f) for f in _FLUXES],
    *[_OrderClaim(FVMDiscretization(f)) for f in _FLUXES],
    # Diffusive (SPD, DirichletBC): all solvers including CG
    *[
        _SolverClaim(s, FVMDiscretization(f, DirichletGhostCells()), _mesh_n8)
        for s in [*_SOLVERS, *_SPD_SOLVERS]
        for f in _DIFFUSIVE_FLUXES
    ],
    *[
        _DirectSolverClaim(s, FVMDiscretization(f, DirichletGhostCells()), _mesh_n8)
        for s in _DIRECT_SOLVERS
        for f in _DIFFUSIVE_FLUXES
    ],
    *[
        _ConvergenceRateClaim(s, FVMDiscretization(f, DirichletGhostCells()))
        for s in [*_SOLVERS, *_SPD_SOLVERS, *_DIRECT_SOLVERS]
        for f in _DIFFUSIVE_FLUXES
    ],
    # Advective (rank-(N-1) circulant, PeriodicBC): direct solver only
    *[
        _DirectSolverClaim(s, FVMDiscretization(f, PeriodicGhostCells()), _mesh_n8)
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES
    ],
    *[
        _ConvergenceRateClaim(s, FVMDiscretization(f, PeriodicGhostCells()))
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES
    ],
    # Advection-diffusion (non-singular under DirichletBC for κ>0): non-SPD solvers only
    *[
        _SolverClaim(s, FVMDiscretization(f, DirichletGhostCells()), _mesh_n8)
        for s in _SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES
    ],
    *[
        _DirectSolverClaim(s, FVMDiscretization(f, DirichletGhostCells()), _mesh_n8)
        for s in _DIRECT_SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES
    ],
    *[
        _ConvergenceRateClaim(s, FVMDiscretization(f, DirichletGhostCells()))
        for s in [*_SOLVERS, *_DIRECT_SOLVERS]
        for f in _ADVECTION_DIFFUSION_FLUXES
    ],
    # FD order claims: verify interior truncation error is O(h^p) for point-value DOFs.
    _OrderClaim(FDDiscretization(2, _fd_cont_op), point_dof=True),
    _OrderClaim(FDDiscretization(4, _fd_cont_op), point_dof=True),
    # FD solver claims: order=2 yields an SPD matrix under DirichletGhostCells.
    *[
        _SolverClaim(s, _FD_DISC_2, _mesh_n8)
        for s in [*_SOLVERS, *_SPD_SOLVERS, *_DIRECT_SOLVERS]
    ],
    # FD convergence rate: order=2 achieves O(h²) globally with ghost-cell BCs.
    # Order≥4 is limited to O(h²) by the one-layer ghost-cell boundary treatment;
    # high-order BCs are not yet implemented.
    *[
        _ConvergenceRateClaim(s, _FD_DISC_2, point_dof=True)
        for s in [*_SOLVERS, *_SPD_SOLVERS, *_DIRECT_SOLVERS]
    ],
]

# Count convergence-rate claims after the list is built so that adding or
# removing claims automatically updates the per-claim time budget.
_N_CONVERGENCE_CLAIMS: int = sum(
    1 for c in _CLAIMS if isinstance(c, _ConvergenceRateClaim)
)


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_convergence(claim: CalibratedClaim[float], fma_rate: float) -> None:
    claim.check(fma_rate)
