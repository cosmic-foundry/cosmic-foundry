"""FDDiscretization: centered finite-difference DiscreteOperator on a CartesianMesh."""

from __future__ import annotations

from typing import cast

import sympy

from cosmic_foundry.geometry.cartesian_mesh import CartesianMesh
from cosmic_foundry.theory.continuous.differential_operator import DifferentialOperator
from cosmic_foundry.theory.discrete.discrete_boundary_condition import (
    DiscreteBoundaryCondition,
    _apply_zero_ghosts,
)
from cosmic_foundry.theory.discrete.discrete_field import (
    DiscreteField,
    _CallableDiscreteField,
)
from cosmic_foundry.theory.discrete.discretization import Discretization


def _fd_laplacian_coeffs(order: int) -> dict[int, sympy.Rational]:
    """Centered FD coefficients for d²/dx² at the given even order.

    Returns {offset: coeff} for the stencil (1/h²) Σ_k c_k u[i+k], with
    offsets in [-order//2, +order//2] and symmetry c_{-k} = c_k.

    Coefficients satisfy the even-moment system:
        c_0 δ_{m,0} + 2 Σ_{k=1}^{p} c_k k^{2m} = 2 δ_{m,1}
    for m = 0, …, p where p = order//2.  This gives p+1 equations for the
    p+1 unknowns c_0, c_1, …, c_p.
    """
    p = order // 2
    c = [sympy.Symbol(f"c{k}") for k in range(p + 1)]
    equations = []
    for m in range(p + 1):
        power = 2 * m
        zero_term = c[0] if m == 0 else sympy.Integer(0)
        pos_terms = 2 * sum(c[k] * sympy.Integer(k) ** power for k in range(1, p + 1))
        rhs = sympy.Integer(2) if m == 1 else sympy.Integer(0)
        equations.append(zero_term + pos_terms - rhs)
    sol = sympy.solve(equations, c)
    coeffs: dict[int, sympy.Rational] = {0: sol[c[0]]}
    for k in range(1, p + 1):
        coeffs[k] = sol[c[k]]
        coeffs[-k] = sol[c[k]]
    return coeffs


class FDDiscretization(Discretization[sympy.Expr]):
    """Centered finite-difference discretization of the scalar Laplacian on a
    CartesianMesh.

    Approximates -∇²φ = -Σ_a ∂²φ/∂x_a² at cell centers to O(h^order) using a
    symmetric stencil derived from even-moment Taylor constraints.  Input DOFs
    are point values at cell centers; ghost-cell BCs extend the field beyond the
    mesh boundary before the stencil is applied.

    The stiffness matrix (extracted by Operator) is symmetric for any order.
    For order=2 with DirichletGhostCells it is positive definite, so CG applies.
    For order≥4 the ghost-cell boundary treatment (one reflected ghost per face)
    achieves O(h²) accuracy at boundary-adjacent cells, which can limit global
    convergence to min(order, 2) even though interior truncation error is O(h^order).
    Specialized high-order boundary stencils are not yet implemented.

    Parameters
    ----------
    mesh:
        CartesianMesh defining the domain geometry and cell spacing.
    order:
        Even integer ≥ 2.  Stencil is accurate to O(h^order) at interior cells.
    continuous_operator:
        The continuous operator this scheme approximates.  Must return a ZeroForm
        when called with a ZeroForm argument.  Typically
        DivergenceComposition(DiffusionOperator(manifold)) for the Laplacian.
    boundary_condition:
        Ghost-cell extension rule; uses zero ghost cells if None.
    """

    min_order: int = 2
    order_step: int = 2

    def __init__(
        self,
        mesh: CartesianMesh,
        order: int,
        continuous_operator: DifferentialOperator,
        boundary_condition: DiscreteBoundaryCondition | None = None,
    ) -> None:
        if order < self.min_order or order % self.order_step != 0:
            raise ValueError(
                f"FDDiscretization requires even order ≥ {self.min_order}, got {order}"
            )
        super().__init__(mesh, boundary_condition)
        self._order = order
        self._continuous_operator = continuous_operator
        self._coeffs = _fd_laplacian_coeffs(order)

    @property
    def order(self) -> int:
        return self._order

    @property
    def continuous_operator(self) -> DifferentialOperator:
        return self._continuous_operator

    def __call__(self, U: DiscreteField[sympy.Expr]) -> DiscreteField[sympy.Expr]:
        """Apply -∇²: (Au)[i] = -Σ_a (Σ_k c_k U[i + k·eₐ]) / hₐ²."""
        mesh = cast(CartesianMesh, U.mesh)
        if self._boundary_condition is not None:
            U = self._boundary_condition.extend(U, mesh)
        else:
            U = _apply_zero_ghosts(U, mesh)

        ndim = len(mesh._shape)
        shape = mesh.shape

        def _to_multi(flat: int) -> tuple[int, ...]:
            result = []
            f = flat
            for a in range(ndim):
                result.append(f % shape[a])
                f //= shape[a]
            return tuple(result)

        residuals: list[sympy.Expr] = []
        for flat_i in range(mesh.n_cells):
            idx = _to_multi(flat_i)
            total: sympy.Expr = sympy.Integer(0)
            for axis in range(ndim):
                h = mesh._spacing[axis]
                axis_sum: sympy.Expr = sympy.Integer(0)
                for offset, coeff in self._coeffs.items():
                    neighbor = list(idx)
                    neighbor[axis] += offset
                    axis_sum = axis_sum + coeff * U(tuple(neighbor))  # type: ignore[arg-type]
                total = total + axis_sum / h**2
            residuals.append(-total)

        residuals_frozen = tuple(residuals)

        def lookup(idx: tuple[int, ...]) -> sympy.Expr:
            flat = 0
            stride = 1
            for a, i in enumerate(idx):
                flat += i * stride
                stride *= shape[a]
            return residuals_frozen[flat]

        return _CallableDiscreteField(mesh, lookup)


__all__ = ["FDDiscretization"]
