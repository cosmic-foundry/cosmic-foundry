"""Discretization ABC."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import sympy

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
        assemble_matrix    — unit-basis assembly of the N^d × N^d stiffness matrix
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

    def assemble_matrix(self) -> sympy.Matrix:
        """Assemble the N^d × N^d stiffness matrix via unit-basis evaluation.

        Row ordering is lexicographic: flat index = Σ_a idx[a] · ∏_{b<a} shape[b],
        so axis 0 varies fastest.  Column j is Lₕ eⱼ evaluated at each cell.
        Any boundary condition must be baked into the operator returned by
        __call__ so that ghost cells are applied correctly.
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

        rows: list[list[sympy.Expr]] = [
            [sympy.Integer(0)] * n_total for _ in range(n_total)
        ]

        for j in range(n_total):
            target = to_multi(j)

            def unit(idx: tuple[int, ...], t: tuple[int, ...] = target) -> sympy.Expr:
                return sympy.Integer(1) if idx == t else sympy.Integer(0)

            e_j: MeshFunction[sympy.Expr] = LazyMeshFunction(self.mesh, unit)
            lh_ej = op(e_j)

            for i in range(n_total):
                rows[i][j] = lh_ej(to_multi(i))  # type: ignore[arg-type]

        return sympy.Matrix(rows)


__all__ = ["Discretization"]
