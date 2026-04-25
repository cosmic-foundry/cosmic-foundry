"""Convergence verification for all concrete DiscreteOperator subclasses and solvers.

Each convergence claim is a _Claim subclass that encodes both what is being
verified and how to verify it.  Adding a new claim requires only appending to
_CLAIMS; the single parametric test covers all entries.

  _OrderClaim(instance)   — instance converges at its declared order p
  _SolverClaim(...)       — solver reaches tol with monotonically decreasing
                            residuals and a tractable iteration count

When adding a new concrete DiscreteOperator subclass, add an _OrderClaim.
When adding a new (solver, discretization) pair, add a _SolverClaim.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import pytest
import sympy

from cosmic_foundry.computation.dense_jacobi_solver import DenseJacobiSolver
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
    """Claim: solver reaches tol with monotonically decreasing residuals.

    On the supplied (discretization, rhs) pair, verifies:
      1. Final residual < tol.
      2. ‖f − Au^k‖_{L²_h} decreases at every step.
      3. Asymptotic convergence rate (geometric mean of last 20 ratios) < 1.
      4. Iteration count ≤ the upper bound implied by that rate and the
         initial residual.
    """

    def __init__(
        self,
        solver: DenseJacobiSolver,
        disc: Any,
        rhs: Any,
        label: str,
    ) -> None:
        self._solver = solver
        self._disc = disc
        self._rhs = rhs
        self._label = label

    @property
    def description(self) -> str:
        return self._label

    def check(self) -> None:
        self._solver.solve(self._disc, self._rhs)
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


_manifold = EuclideanManifold(1)
_dummy_mesh = CartesianMesh(
    origin=(sympy.Integer(0),), spacing=(sympy.Integer(1),), shape=(4,)
)
_mesh_n8 = CartesianMesh(
    origin=(sympy.Rational(0),),
    spacing=(sympy.Rational(1, 8),),
    shape=(8,),
)

_CLAIMS: list[_Claim] = [
    _OrderClaim(DiffusiveFlux(DiffusiveFlux.min_order, _manifold)),
    _OrderClaim(
        DiffusiveFlux(DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold)
    ),
    _OrderClaim(
        FVMDiscretization(
            _dummy_mesh, DiffusiveFlux(DiffusiveFlux.min_order, _manifold)
        )()
    ),
    _OrderClaim(
        FVMDiscretization(
            _dummy_mesh,
            DiffusiveFlux(
                DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold
            ),
        )()
    ),
    _SolverClaim(
        solver=DenseJacobiSolver(tol=1e-8, max_iter=10_000),
        disc=FVMDiscretization(
            _mesh_n8,
            DiffusiveFlux(
                DiffusiveFlux.min_order + DiffusiveFlux.order_step, _manifold
            ),
            DirichletBC(_manifold),
        ),
        rhs=LazyMeshFunction(_mesh_n8, lambda idx: 1.0),
        label="DenseJacobiSolver/DiffusiveFlux(order=4)/N=8",
    ),
]


@pytest.mark.parametrize("claim", _CLAIMS, ids=[c.description for c in _CLAIMS])
def test_convergence(claim: _Claim) -> None:
    claim.check()
