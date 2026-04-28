"""Convergence verification for all concrete DiscreteOperator subclasses and solvers.

Each convergence claim is a CalibratedClaim subclass that encodes both what is
being verified and how to verify it.  Adding a new claim requires only appending
to _CLAIMS; the single parametric test covers all entries.

  _OrderClaim(instance)               — instance achieves O(h^p) at declared p
  _SolverClaim(solver, disc, mesh)    — iterative solver residual < tol
  _DirectSolverClaim(solver, disc, mesh)
                                      — direct solver residual < tol after one
                                        factorization pass
  _ConvergenceRateClaim(solver, disc)
                                      — L²_h error converges at >= O(h^{p-0.1})
                                        over a mesh refinement sequence using a
                                        manufactured solution with exact cell
                                        averages for source and reference field

_FLUXES_1D contains the 1-D NumericalFlux instances used for _OrderClaim (the
stencil-order check is done on a 1-D symbolic mesh; dimension is irrelevant).
Multi-D solver and convergence-rate claims are added alongside 1-D ones; the
disc's manifold determines ndim.

_SOLVERS (iterative, non-SPD-restricted), _SPD_SOLVERS (iterative, SPD-only)
and _DIRECT_SOLVERS (direct) are solver registries.  Diffusive claims use all
three; advection-diffusion claims exclude _SPD_SOLVERS (non-symmetric matrix).
The convergence mesh sequence is computed adaptively at runtime from the
machine's FMA rate; see _convergence_n_max.
"""

from __future__ import annotations

import math
import sys
from itertools import product as iproduct
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
from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ZeroForm,
)
from cosmic_foundry.theory.discrete import (
    DirichletGhostCells,
    DivergenceFormDiscretization,
    PeriodicGhostCells,
)
from cosmic_foundry.theory.discrete.discrete_field import _CallableDiscreteField
from tests.calibration import _MESH_FRACTIONS, _NP_BACKEND, _convergence_n_max
from tests.claims import CalibratedClaim, assemble_linear_op

# ---------------------------------------------------------------------------
# Claim classes
# ---------------------------------------------------------------------------


class _OrderClaim(CalibratedClaim[float]):
    """Claim: discrete operator achieves O(h^p) convergence at order p.

    Verifies that the error polynomial has zeros at h⁰…h^{p-1} and a
    nonzero h^p leading term, using a manufactured polynomial solution
    on a 1-D symbolic mesh.  Input field is cell-averaged; exact reference
    is computed via CartesianVolumeRestriction (cell-average DOF convention).
    """

    def __init__(self, instance: Any) -> None:
        self._instance = instance

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
    with its BC; assembled to a LinearOperator at check time.
    """

    def __init__(self, solver: Any, disc: Any, mesh: CartesianMesh) -> None:
        self._solver = solver
        self._disc = disc
        self._mesh = mesh

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        ndim = len(self._mesh.shape)
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._disc).__name__}(order={self._disc.order})/N={n}/{ndim}D"
        )

    def check(self, fma_rate: float) -> None:
        n = math.prod(self._mesh.shape)
        op = assemble_linear_op(self._disc, self._mesh)
        b = Tensor([1.0] * n, backend=_NP_BACKEND)
        u = self._solver.solve(op, b)
        residual = tensor.norm(b - op.apply(u)).get()
        tol = getattr(self._solver, "_tol", 1e-10)
        assert residual < tol, f"Did not converge: residual {residual:.3e}"


class _DirectSolverClaim(CalibratedClaim[float]):
    """Claim: direct solver residual < tol after one factorization pass.

    disc is pre-built with its BC; assembled to a LinearOperator at check time.
    For PeriodicGhostCells the RHS is a zero-mean product of sines so the system
    is consistent (in the column space of the periodic advection operator).
    Works for any spatial dimensionality.
    """

    def __init__(self, solver: Any, disc: Any, mesh: CartesianMesh) -> None:
        self._solver = solver
        self._disc = disc
        self._mesh = mesh

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        ndim = len(self._mesh.shape)
        periodic = isinstance(self._disc.boundary_condition, PeriodicGhostCells)
        suffix = "/periodic" if periodic else ""
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._disc).__name__}(order={self._disc.order})"
            f"/N={n}/{ndim}D{suffix}"
        )

    def check(self, fma_rate: float) -> None:
        n = math.prod(self._mesh.shape)
        op = assemble_linear_op(self._disc, self._mesh)
        if isinstance(self._disc.boundary_condition, PeriodicGhostCells):
            shape = self._mesh.shape
            ndim = len(shape)

            def _idx(flat: int) -> tuple[int, ...]:
                out = []
                for s in shape:
                    out.append(flat % s)
                    flat //= s
                return tuple(out)

            # Use sum mode sin(2π·(x₁+…+xd)) as RHS.  Tensor-product modes
            # sin(2πx)·sin(2πy) contain Fourier components (k,-k) which are in
            # the null space of v·(∂/∂x+∂/∂y) (eigenvalue ∝ sin(2πk/N)+sin(-2πk/N)=0).
            # The sum mode has only the (1,1,…,1) Fourier component, whose eigenvalue
            # i·v/h·d·sin(2π/N) ≠ 0 for N ≥ 3, placing b safely in the column space.
            # The mean over a full period is exactly zero; no subtraction needed.
            raw = [
                math.sin(
                    2
                    * math.pi
                    * sum(float(self._mesh.coordinate(_idx(i))[k]) for k in range(ndim))
                )
                for i in range(n)
            ]
            b = Tensor(raw)
        else:
            b = Tensor([1.0] * n)
        u = self._solver.solve(op, b)
        residual = tensor.norm(b - op.apply(u)).get()
        assert residual < 1e-10, f"Direct solve residual {residual:.3e} >= 1e-10"


class _ConvergenceRateClaim(CalibratedClaim[float]):
    """Claim: ‖φ_h − φ_exact‖_{L²_h} converges at O(h^p) over a mesh sequence.

    Manufactured solution: tensor-product sinusoidal modes
        φ = ∏_k sin(n_k · π · x_k)
    which vanish at all Dirichlet boundaries and are periodic (even n) for
    PeriodicBC.  Cell averages are computed exactly via iterated antiderivatives
    and the inclusion-exclusion formula; this avoids quadrature error that would
    pollute the convergence-order measurement.

    Admissible modes are those for which ‖A_h φ − ρ‖ / ‖ρ‖ < 10% on the
    coarsest mesh.  Null-space components are projected out of the numerical
    solution before computing L²_h error (handles the advective circulant).

    Mesh sequence: per-axis size scales as n_max^(1/ndim) so total cells and
    assembly cost remain roughly constant across dimensions.
    """

    def __init__(self, solver: Any, disc: Any) -> None:
        self._solver = solver
        self._disc = disc

    @property
    def description(self) -> str:
        periodic = isinstance(self._disc.boundary_condition, PeriodicGhostCells)
        suffix = "/periodic" if periodic else ""
        manifold = self._disc.continuous_operator.manifold
        ndim = len(manifold.atlas[0].symbols)
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._disc).__name__}(order={self._disc.order})/"
            f"convergence_rate{suffix}/{ndim}D"
        )

    def check(self, fma_rate: float) -> None:
        n_max = _convergence_n_max(fma_rate, _N_CONVERGENCE_CLAIMS, self._solver)
        cont_op = self._disc.continuous_operator
        manifold = cont_op.manifold
        symbols = manifold.atlas[0].symbols
        ndim = len(symbols)
        p = self._disc.order
        periodic = isinstance(self._disc.boundary_condition, PeriodicGhostCells)

        # Build mesh sequence. Scale per-axis N as n_max^(1/ndim) to keep total
        # cell count and assembly cost bounded across dimensions.
        #
        # Multi-D periodic advection has a large null space: tensor-product modes
        # sin(n_x·π·x)·sin(n_y·π·y) contain Fourier components (k,-k) which satisfy
        # sin(2πk/N)+sin(-2πk/N)=0, putting them in the null space of v·(∂/∂x+∂/∂y).
        # We instead use "sum modes" sin(2π·k·(x₁+…+xd)) whose only Fourier
        # components are (k,…,k): eigenvalue i·v·h·d·sin(2πk/N)≠0 for k<N/2.
        # In 1-D, tensor-product modes (step=2 for periodicity) work fine.
        use_sum_modes = periodic and ndim > 1
        if use_sum_modes:
            step = 1
            min_cells = p + 1
        elif periodic:
            step = 2
            min_cells = 2 * step
        else:
            step = 1
            min_cells = p + 1
        n_per_axis_max = max(min_cells + 1, int(round(n_max ** (1.0 / ndim))))
        n_per_axes = sorted(
            {
                int(n_per_axis_max * f)
                for f in _MESH_FRACTIONS
                if int(n_per_axis_max * f) >= min_cells
            }
        )
        # Need at least 3 distinct mesh sizes for a meaningful slope estimate.
        if len(n_per_axes) < 3:
            base = min_cells
            n_per_axes = list(range(base, base + 3))

        meshes = [
            CartesianMesh(
                origin=tuple(sympy.Rational(0) for _ in range(ndim)),
                spacing=tuple(sympy.Rational(1, n_k) for _ in range(ndim)),
                shape=tuple(n_k for _ in range(ndim)),
            )
            for n_k in n_per_axes
        ]

        # Pre-build operators and SVD decompositions for null-space extraction.
        assembled: list[tuple[Any, list[Any], float, int]] = []
        for mesh in meshes:
            vol = float(mesh.cell_volume)
            n_cells = math.prod(mesh.shape)
            op_m = assemble_linear_op(self._disc, mesh)
            a_m = _assemble_from_op(op_m, n_cells, _NP_BACKEND)
            decomp = SVDFactorization().factorize(a_m)
            s_vec = decomp.s
            vt = decomp.vt
            null_tol = float(s_vec[0]) * n_cells * sys.float_info.epsilon**0.5
            null_vecs = [vt[j] for j in range(n_cells) if float(s_vec[j]) < null_tol]
            assembled.append((op_m, null_vecs, vol, n_cells))

        def _to_multi(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
            out = []
            for s in shape:
                out.append(flat % s)
                flat //= s
            return tuple(out)

        def _iterated_antideriv(expr: sympy.Expr) -> sympy.Expr:
            """Integrate expr over each coordinate symbol in sequence."""
            result = expr
            for s in symbols:
                result = sympy.integrate(result, s)
            return result

        def _cell_avg(F_lam: Any, mesh: CartesianMesh, idx: tuple[int, ...]) -> float:
            """Exact cell average via iterated-antiderivative inclusion-exclusion.

            For a D-D box [lo_0, hi_0] × … × [lo_{D-1}, hi_{D-1}] and iterated
            antiderivative F (satisfying d^D F / dx_0 … dx_{D-1} = f):

                ∫…∫ f = Σ_{ε∈{0,1}^D} (−1)^{D−Σε} F(lo/hi selected by ε)
            """
            vol = float(mesh.cell_volume)
            lo = [
                float(mesh._origin[k]) + idx[k] * float(mesh._spacing[k])
                for k in range(ndim)
            ]
            hi = [lo[k] + float(mesh._spacing[k]) for k in range(ndim)]
            total = 0.0
            for bits in iproduct([0, 1], repeat=ndim):
                coords = tuple(lo[k] if bits[k] == 0 else hi[k] for k in range(ndim))
                sign = (-1) ** (ndim - sum(bits))
                total += sign * F_lam(*coords)
            return total / vol

        # Build manufactured-solution modes.  Two strategies depending on BC:
        #
        # Dirichlet / 1-D periodic — tensor-product modes ∏_k sin(n_k·π·x_k):
        #   Vanish on all Dirichlet faces; even n_k gives 1-periodicity.
        #
        # Multi-D periodic — sum modes sin(2π·k·(x₁+…+xd)):
        #   1-periodic in every coordinate for any integer k.  Their only Fourier
        #   components (k,…,k) have eigenvalue i·v·h·d·sin(2πk/N) for advective
        #   operators, which is non-zero for 1 ≤ k < N/2.  Tensor-product modes
        #   would include (k,-k,…) components that are null for v·(∂/∂x+∂/∂y),
        #   producing a projected solution with near-zero norm and no convergence.
        #
        # In both cases cell averages are computed exactly via iterated
        # antiderivatives; no quadrature error enters the measurement.
        coarsest_shape = meshes[0].shape
        phi_modes_F: list[tuple[Any, Any]] = []

        if use_sum_modes:
            # Sum modes for multi-D periodic: k = 1, 2, …, k_max_sum.
            # k_max_sum < N_coarsest/2 keeps all modes well inside the non-null
            # subspace (sin(2πk/N) is bounded away from 0 for k < N/2).
            k_max_sum = max(1, coarsest_shape[0] // (2 * p))
            sum_x: sympy.Expr = sum(symbols)  # type: ignore[assignment]
            for k in range(1, k_max_sum + 1):
                phi_expr: sympy.Expr = sympy.sin(2 * k * sympy.pi * sum_x)
                phi_zf = ZeroForm(manifold, phi_expr, tuple(symbols))
                rho_expr = cont_op(phi_zf).expr
                F_phi_sym = _iterated_antideriv(phi_expr)
                F_rho_sym = _iterated_antideriv(rho_expr)
                F_phi_lam = sympy.lambdify(symbols, F_phi_sym, "math")
                F_rho_lam = sympy.lambdify(symbols, F_rho_sym, "math")
                phi_modes_F.append((F_phi_lam, F_rho_lam))
        else:
            # Tensor-product modes for Dirichlet BC or 1-D periodic BC.
            k_max = max(step, coarsest_shape[0] // p)
            mode_range = range(step, k_max + 1, step)
            for mode_ns in iproduct(*[mode_range for _ in range(ndim)]):
                phi_expr = sympy.Integer(1)
                for n_k, x_k in zip(mode_ns, symbols, strict=True):
                    phi_expr = phi_expr * sympy.sin(n_k * sympy.pi * x_k)
                phi_zf = ZeroForm(manifold, phi_expr, tuple(symbols))
                rho_expr = cont_op(phi_zf).expr
                F_phi_sym = _iterated_antideriv(phi_expr)
                F_rho_sym = _iterated_antideriv(rho_expr)
                F_phi_lam = sympy.lambdify(symbols, F_phi_sym, "math")
                F_rho_lam = sympy.lambdify(symbols, F_rho_sym, "math")
                phi_modes_F.append((F_phi_lam, F_rho_lam))

        assert phi_modes_F, "mode_range is empty — check step/k_max configuration"

        errors: list[float] = []
        for mesh, (op_m, null_vecs, vol, n_cells) in zip(
            meshes, assembled, strict=True
        ):
            mesh_shape = mesh.shape

            phi_vals = []
            b_vals = []
            for i in range(n_cells):
                idx = _to_multi(i, mesh_shape)
                phi_sum = sum(_cell_avg(F_phi, mesh, idx) for F_phi, _ in phi_modes_F)
                rho_sum = sum(_cell_avg(F_rho, mesh, idx) for _, F_rho in phi_modes_F)
                phi_vals.append(phi_sum)
                b_vals.append(rho_sum)

            b_m = Tensor(b_vals, backend=_NP_BACKEND)
            phi_arr = Tensor(phi_vals, backend=_NP_BACKEND)

            u_arr = self._solver.solve(op_m, b_m)
            for v in null_vecs:
                u_arr = u_arr - float(u_arr @ v) * v
            diff = u_arr - phi_arr
            errors.append(math.sqrt(vol * (diff @ diff)))

        # Effective mesh spacing: h = vol^(1/ndim) for a uniform Cartesian mesh.
        hs = [float(m.cell_volume) ** (1.0 / ndim) for m in meshes]
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
        # Multi-D periodic (sum-mode) claims use looser thresholds: budget constraints
        # force small meshes (N≈5–10 per axis) that sit in the pre-asymptotic regime
        # where the order-p stencil hasn't fully reached its asymptotic rate.
        # Diagnostics show slope converging from below (e.g. 3.62 at N=4→6 to 3.92
        # at N=12→16 for p=4).  The 1-D periodic tests validate the asymptotic rate
        # precisely; here we check that convergence is occurring at the right order
        # of magnitude — not flat, not regressing — which already catches real holes.
        slope_min = p - 0.35 if use_sum_modes else p - 0.1
        r2_min = 0.99 if use_sum_modes else 0.999
        assert slope >= slope_min, (
            f"Convergence rate {slope:.3f} < expected {slope_min:.2f} for "
            f"{type(self._disc).__name__}(order={p})"
        )
        assert r2 >= r2_min, (
            f"Convergence not clean power-law: R²={r2:.4f} for "
            f"{type(self._disc).__name__}(order={p})"
        )


def _assemble_from_op(op: Any, n: int, backend: Any) -> Any:
    """Build the N×N stiffness matrix from op.apply on basis vectors."""
    columns: list[list[float]] = []
    for j in range(n):
        e_j = Tensor.zeros(n, backend=backend)
        e_j = e_j.set(j, Tensor(1.0, backend=backend))
        columns.append(backend.flatten(op.apply(e_j)._value))
    rows = [[columns[j][i] for j in range(n)] for i in range(n)]
    return Tensor(rows, backend=backend)


# ---------------------------------------------------------------------------
# Registries: manifolds, meshes, fluxes, solvers
# ---------------------------------------------------------------------------

_manifold_1d = EuclideanManifold(1)
_manifold_2d = EuclideanManifold(2)
_manifold_3d = EuclideanManifold(3)

# 1-D solver-claim mesh (8 cells)
_mesh_n8_1d = CartesianMesh(
    origin=(sympy.Rational(0),),
    spacing=(sympy.Rational(1, 8),),
    shape=(8,),
)
# 2-D solver-claim mesh (4×4 = 16 cells)
_mesh_n4_2d = CartesianMesh(
    origin=(sympy.Rational(0), sympy.Rational(0)),
    spacing=(sympy.Rational(1, 4), sympy.Rational(1, 4)),
    shape=(4, 4),
)
# 3-D solver-claim mesh (3×3×3 = 27 cells)
_mesh_n3_3d = CartesianMesh(
    origin=(sympy.Rational(0), sympy.Rational(0), sympy.Rational(0)),
    spacing=(sympy.Rational(1, 3), sympy.Rational(1, 3), sympy.Rational(1, 3)),
    shape=(3, 3, 3),
)

_DIFFUSIVE_FLUXES_1D = [
    DiffusiveFlux(DiffusiveFlux.min_order, _manifold_1d),
    DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold_1d),
]
_ADVECTIVE_FLUXES_1D = [
    AdvectiveFlux(AdvectiveFlux.min_order, _manifold_1d),
    AdvectiveFlux(AdvectiveFlux.min_order + AdvectiveFlux.order_step, _manifold_1d),
]
_ADVECTION_DIFFUSION_FLUXES_1D = [
    AdvectionDiffusionFlux(AdvectionDiffusionFlux.min_order, _manifold_1d),
    AdvectionDiffusionFlux(
        AdvectionDiffusionFlux.min_order + AdvectionDiffusionFlux.order_step,
        _manifold_1d,
    ),
]
# _FLUXES_1D is used only for _OrderClaim (1-D symbolic stencil check).
_FLUXES_1D = [
    *_DIFFUSIVE_FLUXES_1D,
    *_ADVECTIVE_FLUXES_1D,
    *_ADVECTION_DIFFUSION_FLUXES_1D,
]

_DIFFUSIVE_FLUXES_2D = [
    DiffusiveFlux(DiffusiveFlux.min_order, _manifold_2d),
    DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold_2d),
]
_ADVECTIVE_FLUXES_2D = [
    AdvectiveFlux(AdvectiveFlux.min_order, _manifold_2d),
    AdvectiveFlux(AdvectiveFlux.min_order + AdvectiveFlux.order_step, _manifold_2d),
]
_ADVECTION_DIFFUSION_FLUXES_2D = [
    AdvectionDiffusionFlux(AdvectionDiffusionFlux.min_order, _manifold_2d),
    AdvectionDiffusionFlux(
        AdvectionDiffusionFlux.min_order + AdvectionDiffusionFlux.order_step,
        _manifold_2d,
    ),
]

_DIFFUSIVE_FLUXES_3D = [
    DiffusiveFlux(DiffusiveFlux.min_order, _manifold_3d),
    DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold_3d),
]
_ADVECTIVE_FLUXES_3D = [
    AdvectiveFlux(AdvectiveFlux.min_order, _manifold_3d),
    AdvectiveFlux(AdvectiveFlux.min_order + AdvectiveFlux.order_step, _manifold_3d),
]
_ADVECTION_DIFFUSION_FLUXES_3D = [
    AdvectionDiffusionFlux(AdvectionDiffusionFlux.min_order, _manifold_3d),
    AdvectionDiffusionFlux(
        AdvectionDiffusionFlux.min_order + AdvectionDiffusionFlux.order_step,
        _manifold_3d,
    ),
]

# DiffusiveFlux → SPD matrix (DirichletBC): all solvers including CG.
# AdvectiveFlux → rank-(N-1) circulant (PeriodicBC): direct solvers only.
# AdvectionDiffusionFlux → non-symmetric (DirichletBC): no CG.
_SOLVERS = [
    DenseJacobiSolver(tol=1e-8),
    DenseGaussSeidelSolver(tol=1e-8),
    DenseGMRESSolver(tol=1e-8),
]
_SPD_SOLVERS = [DenseCGSolver(tol=1e-8)]
_DIRECT_SOLVERS = [DenseLUSolver(), DenseSVDSolver()]


_CLAIMS: list[CalibratedClaim[float]] = [
    # Order claims: 1-D symbolic stencil test; spatial dimension is irrelevant.
    *[_OrderClaim(f) for f in _FLUXES_1D],
    *[_OrderClaim(DivergenceFormDiscretization(f)) for f in _FLUXES_1D],
    # ---- 1-D solver claims ----
    *[
        _SolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n8_1d
        )
        for s in [*_SOLVERS, *_SPD_SOLVERS]
        for f in _DIFFUSIVE_FLUXES_1D
    ],
    *[
        _DirectSolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n8_1d
        )
        for s in _DIRECT_SOLVERS
        for f in _DIFFUSIVE_FLUXES_1D
    ],
    *[
        _ConvergenceRateClaim(s, DivergenceFormDiscretization(f, DirichletGhostCells()))
        for s in [*_SOLVERS, *_SPD_SOLVERS, *_DIRECT_SOLVERS]
        for f in _DIFFUSIVE_FLUXES_1D
    ],
    *[
        _DirectSolverClaim(
            s, DivergenceFormDiscretization(f, PeriodicGhostCells()), _mesh_n8_1d
        )
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES_1D
    ],
    *[
        _ConvergenceRateClaim(s, DivergenceFormDiscretization(f, PeriodicGhostCells()))
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES_1D
    ],
    *[
        _SolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n8_1d
        )
        for s in _SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES_1D
    ],
    *[
        _DirectSolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n8_1d
        )
        for s in _DIRECT_SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES_1D
    ],
    *[
        _ConvergenceRateClaim(s, DivergenceFormDiscretization(f, DirichletGhostCells()))
        for s in [*_SOLVERS, *_DIRECT_SOLVERS]
        for f in _ADVECTION_DIFFUSION_FLUXES_1D
    ],
    # ---- 2-D solver and convergence-rate claims ----
    *[
        _SolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n4_2d
        )
        for s in [*_SOLVERS, *_SPD_SOLVERS]
        for f in _DIFFUSIVE_FLUXES_2D
    ],
    *[
        _DirectSolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n4_2d
        )
        for s in _DIRECT_SOLVERS
        for f in _DIFFUSIVE_FLUXES_2D
    ],
    *[
        _ConvergenceRateClaim(s, DivergenceFormDiscretization(f, DirichletGhostCells()))
        for s in [*_SOLVERS, *_SPD_SOLVERS, *_DIRECT_SOLVERS]
        for f in _DIFFUSIVE_FLUXES_2D
    ],
    *[
        _DirectSolverClaim(
            s, DivergenceFormDiscretization(f, PeriodicGhostCells()), _mesh_n4_2d
        )
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES_2D
    ],
    *[
        _ConvergenceRateClaim(s, DivergenceFormDiscretization(f, PeriodicGhostCells()))
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES_2D
    ],
    *[
        _SolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n4_2d
        )
        for s in _SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES_2D
    ],
    *[
        _DirectSolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n4_2d
        )
        for s in _DIRECT_SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES_2D
    ],
    *[
        _ConvergenceRateClaim(s, DivergenceFormDiscretization(f, DirichletGhostCells()))
        for s in [*_SOLVERS, *_DIRECT_SOLVERS]
        for f in _ADVECTION_DIFFUSION_FLUXES_2D
    ],
    # ---- 3-D solver claims (assembly + solve correctness; no convergence rate) ----
    *[
        _SolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n3_3d
        )
        for s in [*_SOLVERS, *_SPD_SOLVERS]
        for f in _DIFFUSIVE_FLUXES_3D
    ],
    *[
        _DirectSolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n3_3d
        )
        for s in _DIRECT_SOLVERS
        for f in _DIFFUSIVE_FLUXES_3D
    ],
    *[
        _DirectSolverClaim(
            s, DivergenceFormDiscretization(f, PeriodicGhostCells()), _mesh_n3_3d
        )
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES_3D
    ],
    *[
        _SolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n3_3d
        )
        for s in _SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES_3D
    ],
    *[
        _DirectSolverClaim(
            s, DivergenceFormDiscretization(f, DirichletGhostCells()), _mesh_n3_3d
        )
        for s in _DIRECT_SOLVERS
        for f in _ADVECTION_DIFFUSION_FLUXES_3D
    ],
]

# Count convergence-rate claims so that adding or removing claims automatically
# updates the per-claim time budget in _convergence_n_max.
_N_CONVERGENCE_CLAIMS: int = sum(
    1 for c in _CLAIMS if isinstance(c, _ConvergenceRateClaim)
)


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_convergence(claim: CalibratedClaim[float], fma_rate: float) -> None:
    claim.check(fma_rate)
