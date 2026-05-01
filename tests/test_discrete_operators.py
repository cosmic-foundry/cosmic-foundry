"""Discrete-operator convergence claims.

The module owns two convergence checks:
  _OrderClaim(instance) — symbolic stencil check for the declared order.
  _ConvergenceRateClaim(solver, disc) — manufactured-solution solve whose
      L2_h error must converge at the discretization's declared order.

The manufactured-solution claim uses solvers as execution machinery, but the
asserted convergence rate is the discrete operator's mathematical order. Solver
residual claims live in tests/test_linear_solvers.py.
"""

from __future__ import annotations

import functools
import math
import sys
import time
from itertools import product as iproduct
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation.autotuning.benchmarker import fit_log_log
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.decompositions.svd_factorization import SVDFactorization
from cosmic_foundry.computation.solvers.dense_cg_solver import DenseCGSolver
from cosmic_foundry.computation.solvers.dense_gauss_seidel_solver import (
    DenseGaussSeidelSolver,
)
from cosmic_foundry.computation.solvers.dense_gmres_solver import DenseGMRESSolver
from cosmic_foundry.computation.solvers.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.solvers.dense_lu_solver import DenseLUSolver
from cosmic_foundry.computation.solvers.dense_svd_solver import DenseSVDSolver
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.geometry.cartesian_restriction_operator import (
    CartesianFaceRestriction,
)
from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.continuous.differential_form import (
    DifferentialForm,
    OneForm,
    ZeroForm,
)
from cosmic_foundry.theory.discrete import (
    AdvectionDiffusionFlux,
    AdvectiveFlux,
    DiffusiveFlux,
    DirichletGhostCells,
    DivergenceFormDiscretization,
    PeriodicGhostCells,
)
from cosmic_foundry.theory.discrete.discrete_field import _CallableDiscreteField
from tests.claims import SOLVER_CONVERGENCE_BUDGET_S, Claim

# ---------------------------------------------------------------------------
# Dimension and budget configuration
# ---------------------------------------------------------------------------

# All claim types are generated for each dimension in this list, subject to
# the two documented exceptions in the module docstring.
_DIMS = [1, 2, 3]

# _ConvergenceRateClaim is generated only for dimensions <= this value.
# 3-D sympy assembly over O(N³) cells × O(N³) symbolic variables is O(N^6)
# and prohibitively slow for the mesh-refinement sequences needed to measure
# a convergence rate.
_MAX_CONVERGENCE_RATE_DIM = 2

# Cells per axis for solver-claim meshes; chosen so total cell count (n^ndim)
# stays small enough for sympy assembly to remain fast.
_SOLVER_MESH_N = {1: 8, 2: 4, 3: 3}

# NumpyBackend for convergence claims: numpy SVD/solve are LAPACK-backed and
# the dominant cost is the O(N²) Python-loop assembly, so the cost model fits
# cleanly.  PythonBackend SVD is superlinear at N>64, making calibration noisy.
_NP_BACKEND = NumpyBackend()

# Mesh fractions for each convergence-rate sweep.  Fractions are exact over
# multiples of 8, so all mesh sizes are integers whenever N_max is a multiple
# of 8.
_MESH_FRACTIONS = (0.25, 0.375, 0.5, 0.75, 1.0)

_CALIB_N = 64
_MAX_PROBE_TIME_S = 1.5
_CALIB_MANIFOLD = EuclideanManifold(1)
_ASSEMBLER = DirectSolver(SVDFactorization())


class _DiscreteApplyOperator:
    def __init__(self, disc: Any, mesh: CartesianMesh) -> None:
        self._disc = disc
        self._mesh = mesh
        self._n = mesh.n_cells
        self._shape = mesh.shape

    def _to_flat(self, idx: tuple[int, ...]) -> int:
        flat, stride = 0, 1
        for axis, cell in enumerate(idx):
            flat += cell * stride
            stride *= self._shape[axis]
        return flat

    def _to_multi(self, flat: int) -> tuple[int, ...]:
        idx = []
        for size in self._shape:
            idx.append(flat % size)
            flat //= size
        return tuple(idx)

    def apply(self, u: Tensor) -> Tensor:
        field = _CallableDiscreteField(
            self._mesh, lambda idx: float(u[self._to_flat(idx)])
        )
        result = self._disc(field)
        values = [float(result(self._to_multi(i))) for i in range(self._n)]
        return Tensor(values, backend=u.backend)


class _DiscreteLinearOperator:
    """LinearOperator whose matrix is assembled through DirectSolver._assemble."""

    def __init__(self, disc: Any, mesh: CartesianMesh) -> None:
        self._n = mesh.n_cells
        self._apply_op = _DiscreteApplyOperator(disc, mesh)
        self._matrices: dict[int, Tensor] = {}

    def _matrix(self, backend: Any) -> Tensor:
        key = id(backend)
        if key not in self._matrices:
            self._matrices[key] = _ASSEMBLER._assemble(
                self._apply_op, Tensor.zeros(self._n, backend=backend)
            )
        return self._matrices[key]

    def apply(self, u: Tensor) -> Tensor:
        return self._matrix(u.backend) @ u

    def diagonal(self, backend: Any) -> Tensor:
        matrix = self._matrix(backend)
        return Tensor([float(matrix[i, i]) for i in range(self._n)], backend=backend)

    def row_abs_sums(self, backend: Any) -> Tensor:
        matrix = self._matrix(backend)
        return Tensor(
            [
                sum(abs(float(matrix[i, j])) for j in range(self._n))
                for i in range(self._n)
            ],
            backend=backend,
        )


def _time_solve_at(solver_class: type, n: int) -> float:
    mesh = CartesianMesh(
        origin=(sympy.Rational(0),),
        spacing=(sympy.Rational(1, n),),
        shape=(n,),
    )
    flux = DiffusiveFlux(DiffusiveFlux.min_order, _CALIB_MANIFOLD)
    disc = DivergenceFormDiscretization(flux, DirichletGhostCells())
    b_cal = Tensor([1.0] * n, backend=_NP_BACKEND)
    solver = solver_class()
    op = _DiscreteLinearOperator(disc, mesh)
    solver.solve(op, b_cal)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        op = _DiscreteLinearOperator(disc, mesh)
        solver.solve(op, b_cal)
        best = min(best, time.perf_counter() - t0)
    return best


@functools.cache
def _calibrate_alpha(solver_class: type, fma_rate: float) -> tuple[float, float]:
    """Fit T ~= alpha * N^p / fma_rate for the solver-backed convergence sweep."""
    n2 = _CALIB_N
    wall0 = time.perf_counter()
    t1 = _time_solve_at(solver_class, n2 // 2)
    if time.perf_counter() - wall0 > _MAX_PROBE_TIME_S:
        n2 //= 2
        t1 = _time_solve_at(solver_class, n2 // 2)
    t2 = _time_solve_at(solver_class, n2)
    alpha_raw, exponent = fit_log_log([(n2 // 2, t1), (n2, t2)])
    return alpha_raw * fma_rate, exponent


def _convergence_n_max(fma_rate: float, n_convergence_claims: int, solver: Any) -> int:
    alpha, p = _calibrate_alpha(type(solver), fma_rate)
    sum_fp = sum(f**p for f in _MESH_FRACTIONS)
    budget_per_claim = SOLVER_CONVERGENCE_BUDGET_S / n_convergence_claims
    n_raw = (budget_per_claim * fma_rate / (alpha * sum_fp)) ** (1 / p)
    return max(16, round(n_raw / 8) * 8)


# ---------------------------------------------------------------------------
# Claim classes
# ---------------------------------------------------------------------------


class _OrderClaim(Claim[float]):
    """Claim: discrete operator achieves O(h^p) convergence at order p.

    Verifies that the error polynomial has zeros at h⁰…h^{p-1} and a
    nonzero h^p leading term, using a manufactured polynomial solution
    on a symbolic mesh.

    The test polynomial varies only in the first coordinate (x₁):
        phi(x₁, …, xₙ) = Σ aₖ x₁ᵏ

    In dimension 1 this is the standard 1-D stencil test.  In dimensions 2
    and 3 the same polynomial exercises the full multi-D code path (ghost-cell
    extension, per-axis face flux assembly, divergence sum, cell-volume
    normalization) while keeping the SymPy integration fast: x₂, …, xₙ
    integrals contribute only scalar h factors and simplify immediately.

    Note: this claim is applied to raw NumericalFlux instances only in 1D.
    In d dimensions the raw-flux face error is O(h^{p+d-1}) (face_area = h^{d-1}),
    which would incorrectly report order p+ndim-1.  For ndim > 1 only
    DivergenceFormDiscretization instances are tested here; the face_area and
    vol factors cancel and the cell-average error remains O(h^p).
    """

    def __init__(self, instance: Any) -> None:
        self._instance = instance

    @property
    def description(self) -> str:
        manifold = self._instance.continuous_operator.manifold
        ndim = len(manifold.atlas[0].symbols)
        return f"{type(self._instance).__name__}(order={self._instance.order})/{ndim}D"

    def check(self, fma_rate: float) -> None:
        instance = self._instance
        h = sympy.Symbol("h", positive=True)
        order = instance.order
        n = order // 2

        manifold = instance.continuous_operator.manifold
        symbols = manifold.atlas[0].symbols
        ndim = len(symbols)
        x = symbols[0]
        mesh = CartesianMesh(
            origin=tuple(sympy.Integer(0) for _ in range(ndim)),
            spacing=tuple(h for _ in range(ndim)),
            shape=tuple(2 * n + 2 for _ in range(ndim)),
        )

        coeffs = sympy.symbols(f"a:{order + 4}")
        phi_expr: sympy.Expr = sum(c * x**k for k, c in enumerate(coeffs))
        phi = ZeroForm(manifold, phi_expr, tuple(symbols))

        # Compute cell averages lazily (on demand) rather than eagerly over all
        # mesh cells.  CartesianVolumeRestriction iterates product(*shape) cells
        # upfront; in 3D with shape (6,6,6) that is 216 SymPy integrations before
        # the stencil is even evaluated.  The stencil at center only reads O(p·ndim)
        # cells, so lazy evaluation reduces the work by ~10–50×.
        vol = mesh.cell_volume

        def _integrate_over_cell(expr: sympy.Expr, idx: tuple[int, ...]) -> sympy.Expr:
            """Exact cell average of expr at cell idx via iterated integration."""
            e = expr
            for k, sym in enumerate(symbols):
                lo = mesh._origin[k] + sympy.Integer(idx[k]) * mesh._spacing[k]
                hi = lo + mesh._spacing[k]
                e = sympy.integrate(e, (sym, lo, hi))
            return sympy.simplify(e) / vol

        # Cache cell averages of phi_expr for U_avg (stencil reads O(p·ndim) cells).
        _phi_avg_cache: dict[tuple[int, ...], sympy.Expr] = {}

        def _phi_avg(idx: tuple[int, ...]) -> sympy.Expr:
            if idx not in _phi_avg_cache:
                _phi_avg_cache[idx] = _integrate_over_cell(phi_expr, idx)
            return _phi_avg_cache[idx]

        U_avg = _CallableDiscreteField(mesh, _phi_avg)
        numerical_mf = instance(U_avg)
        cont_result = instance.continuous_operator(phi)
        assert isinstance(cont_result, DifferentialForm)
        center: tuple[int, ...] = tuple(n for _ in range(ndim))
        if isinstance(cont_result, ZeroForm):
            test_idx: Any = center
            # Only one exact value needed; no cache required.
            error = sympy.expand(
                sympy.simplify(
                    numerical_mf(test_idx)
                    - _integrate_over_cell(cont_result.expr, test_idx)
                )
            )
        else:
            assert isinstance(cont_result, OneForm)
            exact_mf = CartesianFaceRestriction(mesh)(cont_result)
            test_idx = (0, center)
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


class _ConvergenceRateClaim(Claim[float]):
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
            op_m = _DiscreteLinearOperator(self._disc, mesh)
            a_m = _ASSEMBLER._assemble(op_m, Tensor.zeros(n_cells, backend=_NP_BACKEND))
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


# ---------------------------------------------------------------------------
# Solver registries
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Claim generation: discrete-operator order and manufactured-solution convergence
# ---------------------------------------------------------------------------

_CLAIMS: list[Claim[float]] = []

for _ndim in _DIMS:
    _manifold = EuclideanManifold(_ndim)

    _diff_fluxes = [
        DiffusiveFlux(DiffusiveFlux.min_order, _manifold),
        DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold),
    ]
    _adv_fluxes = [
        AdvectiveFlux(AdvectiveFlux.min_order, _manifold),
        AdvectiveFlux(AdvectiveFlux.min_order + AdvectiveFlux.order_step, _manifold),
    ]
    _adv_diff_fluxes = [
        AdvectionDiffusionFlux(AdvectionDiffusionFlux.min_order, _manifold),
        AdvectionDiffusionFlux(
            AdvectionDiffusionFlux.min_order + AdvectionDiffusionFlux.order_step,
            _manifold,
        ),
    ]

    if _ndim == 1:
        for _f in [*_diff_fluxes, *_adv_fluxes, *_adv_diff_fluxes]:
            _CLAIMS.append(_OrderClaim(_f))
    for _f in [*_diff_fluxes, *_adv_fluxes, *_adv_diff_fluxes]:
        _CLAIMS.append(_OrderClaim(DivergenceFormDiscretization(_f)))

    if _ndim <= _MAX_CONVERGENCE_RATE_DIM:
        for _f in _diff_fluxes:
            _disc = DivergenceFormDiscretization(_f, DirichletGhostCells())
            for _s in [*_SOLVERS, *_SPD_SOLVERS, *_DIRECT_SOLVERS]:
                _CLAIMS.append(_ConvergenceRateClaim(_s, _disc))
        for _f in _adv_fluxes:
            _disc = DivergenceFormDiscretization(_f, PeriodicGhostCells())
            for _s in _DIRECT_SOLVERS:
                _CLAIMS.append(_ConvergenceRateClaim(_s, _disc))
        for _f in _adv_diff_fluxes:
            _disc = DivergenceFormDiscretization(_f, DirichletGhostCells())
            for _s in [*_SOLVERS, *_DIRECT_SOLVERS]:
                _CLAIMS.append(_ConvergenceRateClaim(_s, _disc))

_N_CONVERGENCE_CLAIMS: int = sum(
    1 for c in _CLAIMS if isinstance(c, _ConvergenceRateClaim)
)


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_convergence(claim: Claim[float], fma_rate: float) -> None:
    claim.check(fma_rate)
