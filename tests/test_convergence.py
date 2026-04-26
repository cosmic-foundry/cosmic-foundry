"""Convergence verification for all concrete DiscreteOperator subclasses and solvers.

Each convergence claim is a CalibratedClaim subclass that encodes both what is
being verified and how to verify it.

  _OrderClaim(instance)               — instance achieves O(h^p) at declared p
  _SolverClaim(solver, flux, mesh)    — iterative solver residual < tol (CPU)
  _DirectSolverClaim(solver, flux, mesh)
                                      — direct solver residual < tol after one
                                        factorization pass (CPU)
  _ConvergenceRateClaim(solver, flux, device="cpu"|"gpu")
                                      — L²_h error converges at >= O(h^{p-0.1})
                                        over an adaptive mesh sequence; device
                                        selects which backend and time budget
                                        to use

Two registries drive two parametrised test functions:
  _STATIC_CLAIMS  — order, solver, and direct-solver claims (_CLAIMS fixture)
  _RATE_CLAIMS    — convergence-rate claims for each device (convergence_calibration)

_FLUXES contains all NumericalFlux instances; adding a new flux automatically
generates _OrderClaim entries for it.  _SOLVERS (iterative) and _DIRECT_SOLVERS
(direct) are registries for the elliptic claims.  The convergence mesh sequence
length is computed adaptively from Jacobi cost coefficients calibrated per
device; see _convergence_n_max.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.dense_lu_solver import DenseLUSolver
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianRestrictionOperator,
)
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.physics.advection_diffusion_flux import AdvectionDiffusionFlux
from cosmic_foundry.physics.advective_flux import AdvectiveFlux
from cosmic_foundry.physics.diffusive_flux import DiffusiveFlux
from cosmic_foundry.physics.fvm_discretization import FVMDiscretization
from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ThreeForm,
    ZeroForm,
)
from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DirichletGhostCells,
    PeriodicGhostCells,
)
from cosmic_foundry.theory.discrete.discrete_field import _CallableDiscreteField
from tests.claims import (
    BUDGET_TOLERANCE,
    MAX_WALLTIME_S,
    CalibratedClaim,
    ConvergenceCalibration,
)

# ---------------------------------------------------------------------------
# Adaptive mesh-size selection
# ---------------------------------------------------------------------------

# MAX_WALLTIME_S (defined in tests/claims.py) is split across all convergence-
# rate claims.  When both CPU and GPU are present each device gets half the
# budget, so each device's N_max is computed from MAX_WALLTIME_S / 2 /
# n_claims_per_device.  When only CPU is available it receives the full budget.
# BUDGET_TOLERANCE is the over-budget factor enforced at the end of each
# claim's check() — the same factor pytest's session_timeout uses as backstop.

# Mesh fractions: each convergence-rate claim sweeps N_max × f for f in this
# tuple.  The fractions are exact rationals over multiples of 8, so every
# mesh size is an integer whenever N_max is a multiple of 8.
_MESH_FRACTIONS = (0.25, 0.375, 0.5, 0.75, 1.0)


def _convergence_n_max(alpha: float, exponent: int, budget_s: float) -> int:
    """N_max for the convergence mesh sequence for one (device, solver) pair.

    Solves T ≈ alpha × N_max^p × Σ(f^p) = budget_s for N_max, where p is the
    solver's cost_exponent and alpha was calibrated on the target device.
    Rounding to the nearest multiple of 8 keeps all mesh sizes exact integers.

    Solvers with smaller p (LU, p=3) get larger N_max for the same budget than
    solvers with larger p (Jacobi, p=4) — exactly the behavior we want for
    stress-testing convergence at the high-N end.
    """
    sum_fp = sum(f**exponent for f in _MESH_FRACTIONS)
    n_raw = (budget_s / (alpha * sum_fp)) ** (1.0 / exponent)
    return max(16, round(n_raw / 8) * 8)


# ---------------------------------------------------------------------------
# Claim classes
# ---------------------------------------------------------------------------


class _OrderClaim(CalibratedClaim[float]):
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
        ndim = len(mesh._shape)

        coeffs = sympy.symbols(f"a:{order + 4}")
        phi_expr: sympy.Expr = sum(c * x**k for k, c in enumerate(coeffs))
        phi = ZeroForm(space, phi_expr, (x,))

        vol = mesh.cell_volume
        U_totals = CartesianRestrictionOperator(mesh, degree=ndim)(
            _as_n_form(phi, ndim)
        )
        U_avg = _CallableDiscreteField(mesh, lambda idx, _U=U_totals: _U(idx) / vol)
        numerical_mf = instance(U_avg)

        cont_result = instance.continuous_operator(phi)
        assert isinstance(cont_result, DifferentialForm)
        restriction_degree = ndim - cont_result.degree
        if restriction_degree == ndim:
            # cont_result is ZeroForm; wrap as n-form for de Rham-correct restriction
            cont_form: DifferentialForm = _as_n_form(cont_result, ndim)
        else:
            cont_form = cont_result
        exact_mf = CartesianRestrictionOperator(mesh, degree=restriction_degree)(
            cont_form
        )

        test_idx: Any = (0, (n,)) if restriction_degree < ndim else (n,)
        # When restriction_degree==ndim, exact_mf is a VolumeField (totals);
        # numerical_mf is a State (averages).  Normalize exact by vol to compare.
        if restriction_degree == ndim:
            error = sympy.expand(
                sympy.simplify(numerical_mf(test_idx) - exact_mf(test_idx) / vol)
            )
        else:
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
    """Claim: solver converges on FVMDiscretization(mesh, flux, DirichletGhostCells).

    Verifies that ‖b − Au‖₂ < tol after solve returns.
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

    def check(self, fma_rate: float) -> None:
        disc = FVMDiscretization(self._mesh, self._flux, DirichletGhostCells())
        n = math.prod(self._mesh.shape)
        a = disc.assemble()
        b = Tensor([1.0] * n)
        u = self._solver.solve(a, b)
        residual = (b - a @ u).norm().get()
        assert (
            residual < self._solver._tol
        ), f"Did not converge: residual {residual:.3e}"


class _DirectSolverClaim(CalibratedClaim[float]):
    """Claim: direct solver residual < tol after one factorization pass.

    Builds the discretization from flux, mesh, and bc_type, then verifies:
      1. Final residual ‖f − Au‖_{L²_h} < tol.
    No monotonicity or iteration-count checks: a direct solver produces the
    exact solution (up to floating-point rounding) in a single pass.

    For PeriodicGhostCells the RHS is a zero-mean sinusoid so the system is
    consistent (in the column space of the circulant advection matrix).
    """

    def __init__(
        self,
        solver: DenseLUSolver,
        flux: Any,
        mesh: CartesianMesh,
        bc_type: type = DirichletGhostCells,
    ) -> None:
        self._solver = solver
        self._flux = flux
        self._mesh = mesh
        self._bc_type = bc_type

    @property
    def description(self) -> str:
        n = math.prod(self._mesh.shape)
        suffix = "/periodic" if self._bc_type is PeriodicGhostCells else ""
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._flux).__name__}(order={self._flux.order})"
            f"/N={n}{suffix}"
        )

    def check(self, fma_rate: float) -> None:
        bc = self._bc_type()
        disc = FVMDiscretization(self._mesh, self._flux, bc)
        n = math.prod(self._mesh.shape)
        a = disc.assemble()
        if self._bc_type is PeriodicGhostCells:
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
        u = self._solver.solve(a, b)
        residual = (b - a @ u).norm().get()
        assert residual < 1e-10, f"Direct solve residual {residual:.3e} >= 1e-10"


class _ConvergenceRateClaim(CalibratedClaim[ConvergenceCalibration]):
    """Claim: ‖φ_h − Rₕ φ_exact‖_{L²_h} converges at O(h^p) over the mesh sequence.

    The manufactured solution φ is selected automatically from candidates
    sin(nπx) for n=1..k_max, where k_max = N_min // p gives at least 2p
    cells per wavelength on the coarsest mesh (sufficient for the asymptotic
    regime).  A candidate is admitted only when A·R_h(sin(nπx)) matches
    R_h(∇·F(sin(nπx))) to within 10% on the coarsest mesh; modes inconsistent
    with the BC's ghost-cell convention (e.g. odd-n modes under PeriodicGhostCells)
    produce O(1) relative error and are excluded automatically.  The source
    ρ = ∇·F(φ) is derived symbolically from flux.continuous_operator.

    Before measuring the L²_h error the assembled stiffness matrix is
    decomposed via SVD; any null-space components of u_h are projected out,
    isolating truncation error from the arbitrary null-space component a
    direct solver may introduce for singular systems (e.g. advection under
    PeriodicBC).

    device selects the compute target and time budget:
      "cpu" — runs on calibration.cpu_backend; gets half the total budget
              when GPU is also available, else the full budget.
      "gpu" — runs on calibration.gpu_backend; always gets half the total
              budget; auto-skipped when no GPU is available.

    Running small-N claims on CPU and large-N claims on GPU exercises both
    the coarse-grid and fine-grid convergence regimes within the same session.
    """

    def __init__(
        self,
        solver: Any,
        flux: Any,
        bc_type: type = DirichletGhostCells,
        device: str = "cpu",
    ) -> None:
        self._solver = solver
        self._flux = flux
        self._bc_type = bc_type
        self._device = device

    @property
    def description(self) -> str:
        suffix = "/periodic" if self._bc_type is PeriodicGhostCells else ""
        return (
            f"{type(self._solver).__name__}/"
            f"{type(self._flux).__name__}(order={self._flux.order})/"
            f"convergence_rate{suffix}/{self._device}"
        )

    def check(self, calibration: ConvergenceCalibration) -> None:
        t_start = time.perf_counter()
        if self._device == "gpu":
            if calibration.gpu_backend is None:
                pytest.skip("no GPU device available")
            backend = calibration.gpu_backend
            alpha = calibration.gpu_alphas[type(self._solver)]
            budget_s = MAX_WALLTIME_S / 2 / _N_CONVERGENCE_CLAIMS
        else:
            backend = calibration.cpu_backend
            alpha = calibration.cpu_alphas[type(self._solver)]
            has_gpu = calibration.gpu_backend is not None
            pool = MAX_WALLTIME_S / 2 if has_gpu else MAX_WALLTIME_S
            budget_s = pool / _N_CONVERGENCE_CLAIMS

        n_max = _convergence_n_max(alpha, self._solver.cost_exponent, budget_s)
        meshes = [
            CartesianMesh(
                origin=(sympy.Rational(0),),
                spacing=(sympy.Rational(1, int(n_max * f)),),
                shape=(int(n_max * f),),
            )
            for f in _MESH_FRACTIONS
        ]

        manifold = self._flux.continuous_operator.manifold
        _x = manifold.atlas[0].symbols[0]
        p = self._flux.order
        bc = self._bc_type()

        # Auto-select admissible manufactured-solution modes.
        # k_max = N_min // p ensures >= 2p cells/wavelength on the coarsest mesh.
        # Each candidate sin(nπx) is tested for BC consistency: if
        # ||A·R_h(φ_n) - R_h(ρ_n)|| / ||R_h(ρ_n)|| >= 0.1 the mode violates
        # the ghost-cell convention (O(1) boundary error) and is dropped.
        coarse = meshes[0]  # fractions are ascending; first entry is smallest
        n_c = coarse.shape[0]
        vol_c = float(coarse.cell_volume)
        orig_c = float(coarse.coordinate((0,))[0]) - 0.5 * vol_c
        a_c = FVMDiscretization(coarse, self._flux, bc).assemble(backend=backend)
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
            v_n = Tensor(
                [
                    (F_pn(orig_c + (i + 1) * vol_c) - F_pn(orig_c + i * vol_c)) / vol_c
                    for i in range(n_c)
                ],
                backend=backend,
            )
            r_n = Tensor(
                [
                    (F_rn(orig_c + (i + 1) * vol_c) - F_rn(orig_c + i * vol_c)) / vol_c
                    for i in range(n_c)
                ],
                backend=backend,
            )
            rel_err = (a_c @ v_n - r_n).norm().get() / (r_n.norm().get() + 1e-30)
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
        for mesh in meshes:
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
            a_m = disc.assemble(backend=backend)
            _, s_vec, vt = a_m.svd()
            null_tol = float(s_vec[0]) * n_cells * sys.float_info.epsilon**0.5
            null_vecs = [vt[j] for j in range(n_cells) if float(s_vec[j]) < null_tol]

            b_m = Tensor([_rho_avg(i) for i in range(n_cells)], backend=backend)
            u_arr = self._solver.solve(a_m, b_m)
            for v in null_vecs:
                u_arr = u_arr - float(u_arr @ v) * v
            phi_arr = Tensor([_phi_avg(i) for i in range(n_cells)], backend=backend)
            diff = u_arr - phi_arr
            errors.append(math.sqrt(vol * float((diff @ diff).get())))

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
        assert slope >= self._flux.order - 0.1, (
            f"Convergence rate {slope:.3f} < expected "
            f"{self._flux.order - 0.1:.1f} for "
            f"{type(self._flux).__name__}(order={self._flux.order})"
        )
        assert r2 >= 0.999, (
            f"Convergence not clean power-law: R²={r2:.4f} for "
            f"{type(self._flux).__name__}(order={self._flux.order}) — "
            f"errors do not lie on h^p even though the slope is correct"
        )
        elapsed = time.perf_counter() - t_start
        assert elapsed <= budget_s * BUDGET_TOLERANCE, (
            f"{self.description}: took {elapsed:.2f}s, "
            f"budget {budget_s:.2f}s × {BUDGET_TOLERANCE} tolerance "
            f"({elapsed / budget_s:.2f}× budget)"
        )


def _as_n_form(f: ZeroForm, ndim: int) -> DifferentialForm:
    """Wrap scalar density ZeroForm as the n-form f·dV (Cartesian coordinates)."""
    if ndim == 1:
        return OneForm(f.manifold, (f.expr,), f.symbols)
    if ndim == 3:
        return ThreeForm(f.manifold, f.expr, f.symbols)
    raise NotImplementedError(f"_as_n_form not implemented for ndim={ndim}")


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
# DiffusiveFlux assembles an SPD matrix (DirichletGhostCells);
# compatible with all solvers.
# AdvectiveFlux assembles a rank-(N-1) circulant matrix under PeriodicGhostCells;
# compatible with DenseLUSolver only (zero-mean null-space convention handles the
# singularity).
# AdvectionDiffusionFlux assembles A_adv + κ·A_diff; non-singular under
# DirichletGhostCells for any κ > 0, compatible with all solvers at unit Péclet
# number (κ=1, h≈1/N).
_SOLVERS = [DenseJacobiSolver(tol=1e-8)]
_DIRECT_SOLVERS = [DenseLUSolver()]


def _make_rate_claims(device: str) -> list[_ConvergenceRateClaim]:
    return [
        # Diffusive (SPD, DirichletBC): all solvers
        *[
            _ConvergenceRateClaim(s, f, device=device)
            for s in [*_SOLVERS, *_DIRECT_SOLVERS]
            for f in _DIFFUSIVE_FLUXES
        ],
        # Advective (rank-(N-1) circulant, PeriodicBC): direct solver only
        *[
            _ConvergenceRateClaim(s, f, PeriodicGhostCells, device=device)
            for s in _DIRECT_SOLVERS
            for f in _ADVECTIVE_FLUXES
        ],
        # Advection-diffusion (non-singular under DirichletBC for κ>0): all solvers
        *[
            _ConvergenceRateClaim(s, f, device=device)
            for s in [*_SOLVERS, *_DIRECT_SOLVERS]
            for f in _ADVECTION_DIFFUSION_FLUXES
        ],
    ]


_STATIC_CLAIMS: list[CalibratedClaim[float]] = [
    *[_OrderClaim(f) for f in _FLUXES],
    *[_OrderClaim(FVMDiscretization(_dummy_mesh, f)()) for f in _FLUXES],
    # Diffusive (SPD, DirichletBC): all solvers — CPU only, fixed N=8
    *[_SolverClaim(s, f, _mesh_n8) for s in _SOLVERS for f in _DIFFUSIVE_FLUXES],
    *[
        _DirectSolverClaim(s, f, _mesh_n8)
        for s in _DIRECT_SOLVERS
        for f in _DIFFUSIVE_FLUXES
    ],
    # Advective (rank-(N-1) circulant, PeriodicBC): direct solver only — CPU, N=8
    *[
        _DirectSolverClaim(s, f, _mesh_n8, PeriodicGhostCells)
        for s in _DIRECT_SOLVERS
        for f in _ADVECTIVE_FLUXES
    ],
    # Advection-diffusion (κ>0): all solvers, DirichletBC, N=8
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
]

_cpu_rate_claims = _make_rate_claims("cpu")
_gpu_rate_claims = _make_rate_claims("gpu")
_RATE_CLAIMS: list[CalibratedClaim[ConvergenceCalibration]] = [
    *_cpu_rate_claims,
    *_gpu_rate_claims,
]

# Number of rate claims per device — used inside check() to divide the pool
# budget equally across all claims on that device.
_N_CONVERGENCE_CLAIMS: int = len(_cpu_rate_claims)


@pytest.mark.parametrize(
    "claim", _STATIC_CLAIMS, ids=[c.description for c in _STATIC_CLAIMS]
)
def test_convergence(claim: CalibratedClaim[float], fma_rate: float) -> None:
    claim.check(fma_rate)


@pytest.mark.parametrize(
    "claim", _RATE_CLAIMS, ids=[c.description for c in _RATE_CLAIMS]
)
def test_convergence_rate(
    claim: CalibratedClaim[ConvergenceCalibration],
    convergence_calibration: ConvergenceCalibration,
) -> None:
    claim.check(convergence_calibration)
