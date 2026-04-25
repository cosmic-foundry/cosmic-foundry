"""Discretization ABC."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.theory.continuous.boundary_condition import BoundaryCondition
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.lazy_mesh_function import LazyMeshFunction
from cosmic_foundry.theory.discrete.mesh import Mesh
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction


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
        diagonal           — diagonal of the stiffness matrix as a Tensor
        apply              — apply Lₕ to a discrete field; returns a Tensor
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
        n_total = math.prod(shape)

        def to_multi(flat: int) -> tuple[int, ...]:
            idx = []
            for a in range(ndim):
                idx.append(flat % shape[a])
                flat //= shape[a]
            return tuple(idx)

        rows: list[list[float]] = [[0.0] * n_total for _ in range(n_total)]

        for j in range(n_total):
            target = to_multi(j)

            def unit(idx: tuple[int, ...], t: tuple[int, ...] = target) -> sympy.Expr:
                return sympy.Integer(1) if idx == t else sympy.Integer(0)

            e_j: MeshFunction[sympy.Expr] = LazyMeshFunction(self.mesh, unit)
            lh_ej = op(e_j)

            for i in range(n_total):
                rows[i][j] = float(lh_ej(to_multi(i)))  # type: ignore[arg-type]

        return Tensor(rows)

    def diagonal(self) -> Tensor:
        """Diagonal of the assembled stiffness matrix as a rank-1 Tensor."""
        return self.assemble().diag()

    def apply(self, u: Any) -> Tensor:
        """Apply Lₕ to discrete field u; return the result as a rank-1 Tensor.

        u must be indexable with N^d float values in lexicographic
        (axis-0-fastest) order.  The default materialises the full stiffness
        matrix and performs a dense matrix-vector product — O(N^{2d}) memory.
        Override this method for matrix-free implementations.
        """
        a = self.assemble()
        n = a.shape[0]
        u_vec = Tensor([float(u[j]) for j in range(n)])
        result = a @ u_vec
        assert isinstance(result, Tensor)
        return result


__all__ = ["Discretization"]
