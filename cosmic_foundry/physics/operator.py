"""Operator: physics-layer Tensor-backed linear operator on State objects."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.physics.state import State
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


class Operator:
    """Physics-layer Tensor-backed linear operator.

    The materialized form of a discrete linear map: a mesh-bound N×N Tensor
    produced by Operator.assemble() from a symbolic DiscreteOperator and
    consumed directly by LinearSolver.solve(op.matrix, b).

    Parameters
    ----------
    matrix:
        Assembled stiffness Tensor of shape (n_cells, n_cells).
    mesh:
        The mesh on which the operator is defined.
    """

    def __init__(self, matrix: Tensor, mesh: Mesh) -> None:
        self._matrix = matrix
        self._mesh = mesh

    @property
    def matrix(self) -> Tensor:
        """The assembled stiffness matrix."""
        return self._matrix

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, state: State) -> State:
        """Apply the operator to a State via matrix–vector multiply."""
        return State(self._mesh, self._matrix @ state.data)

    @classmethod
    def assemble(
        cls,
        op: DiscreteOperator[Any],
        mesh: Mesh,
        backend: Any = None,
    ) -> Operator:
        """Materialize a symbolic DiscreteOperator into a Tensor-backed Operator.

        Applies op to each sympy unit-basis vector in turn and converts the
        resulting sympy expressions to float.  Row and column ordering is
        axis-0-fastest.  This is the abstract-to-concrete transition: op lives
        in theory/ and is sympy-parameterized; the returned Operator lives in
        physics/ and is Tensor-backed.

        Parameters
        ----------
        op:
            Symbolic DiscreteOperator to materialize.
        mesh:
            The mesh op is defined on; determines matrix dimension n_cells².
        backend:
            Forwarded to Tensor so the assembled matrix lives on the requested
            device (default: process-wide default backend).
        """
        shape = mesh.shape
        ndim = len(shape)
        n_total = mesh.n_cells

        def to_multi(flat: int) -> tuple[int, ...]:
            idx = []
            for a in range(ndim):
                idx.append(flat % shape[a])
                flat //= shape[a]
            return tuple(idx)

        rows: list[list[float]] = [[0.0] * n_total for _ in range(n_total)]

        for j in range(n_total):
            target = to_multi(j)
            e_j = _BasisField(mesh, target)
            lh_ej = op(e_j)
            for i in range(n_total):
                rows[i][j] = float(lh_ej(to_multi(i)))  # type: ignore[arg-type]

        return cls(Tensor(rows, backend=backend), mesh)


__all__ = ["Operator"]
