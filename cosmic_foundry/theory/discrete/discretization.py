"""Discretization ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.theory.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.mesh import Mesh


class _BasisField(DiscreteField[sympy.Expr]):
    """Unit basis vector eⱼ: returns 1 at target cell, 0 everywhere else."""

    def __init__(self, mesh: Mesh, target: tuple[int, ...]) -> None:
        self._mesh = mesh
        self._target = target

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, idx: tuple[int, ...]) -> sympy.Expr:  # type: ignore[override]
        return sympy.Integer(1) if idx == self._target else sympy.Integer(0)


class Discretization(ABC):
    """Encapsulates a discrete scheme on a mesh.

    A Discretization holds the scheme choice — reconstruction, numerical
    flux, quadrature — for a particular mesh and approximation order.
    Calling it produces the DiscreteOperator Lₕ that makes the commutation
    diagram

        Lₕ ∘ Rₕ ≈ Rₕ ∘ L   (up to O(hᵖ))

    hold, where p is the approximation order.

    Required:
        __call__ — produce the DiscreteOperator (signature defined by subclass)

    Concrete:
        mesh               — the mesh on which the scheme is defined
        boundary_condition — the BoundaryCondition on ∂Ω (None if not yet set)
        assemble           — stiffness matrix as a Tensor
    """

    def __init__(
        self,
        mesh: Mesh,
        boundary_condition: BoundaryCondition | None = None,
    ) -> None:
        self._mesh = mesh
        self._boundary_condition = boundary_condition

    @property
    def mesh(self) -> Mesh:
        """The mesh on which the scheme is defined."""
        return self._mesh

    @property
    def boundary_condition(self) -> BoundaryCondition | None:
        """The boundary condition on ∂Ω."""
        return self._boundary_condition

    @abstractmethod
    def __call__(self) -> DiscreteOperator:
        """Produce the assembled DiscreteOperator."""

    def assemble(self) -> Tensor:
        """Assemble the N^d × N^d stiffness matrix as a rank-2 Tensor.

        Applies Lₕ to each sympy unit-basis vector in turn and converts the
        resulting sympy expressions to float.  Row and column ordering is
        lexicographic with axis 0 varying fastest.  Intended for direct
        solvers and inspection; not for large N.
        """
        op = self()
        shape = self.mesh.shape
        ndim = len(shape)
        n_total = self.mesh.n_cells

        def to_multi(flat: int) -> tuple[int, ...]:
            idx = []
            for a in range(ndim):
                idx.append(flat % shape[a])
                flat //= shape[a]
            return tuple(idx)

        rows: list[list[float]] = [[0.0] * n_total for _ in range(n_total)]

        for j in range(n_total):
            target = to_multi(j)

            e_j: DiscreteField[sympy.Expr] = _BasisField(self.mesh, target)
            lh_ej = op(e_j)

            for i in range(n_total):
                rows[i][j] = float(lh_ej(to_multi(i)))  # type: ignore[arg-type]

        return Tensor(rows)


__all__ = ["Discretization"]
