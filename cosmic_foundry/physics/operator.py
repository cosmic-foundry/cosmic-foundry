"""Operator: physics-layer discrete operator instantiated on a Mesh."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.physics.state import State
from cosmic_foundry.theory.discrete.discrete_field import DiscreteField
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.mesh import Mesh


def _to_multi(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    idx = []
    for n in shape:
        idx.append(flat % n)
        flat //= n
    return tuple(idx)


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
    """A symbolic DiscreteOperator instantiated on a specific Mesh.

    Operator binds an abstract scheme (a DiscreteOperator produced by a
    Discretization) to concrete geometry (a Mesh), making it ready to apply
    or assemble:

        op = Operator(disc(), mesh)
        residual = op(state)          # functional apply: O(N) stencil evaluation
        A = op.assemble(backend)      # materialize N×N matrix for a linear solver

    __call__ is the primary operation for time integration; assemble() is the
    cold path used by implicit solvers.  The matrix is not stored — it is
    produced on demand.

    Parameters
    ----------
    op:
        Symbolic DiscreteOperator describing the scheme.
    mesh:
        The Mesh on which the operator is instantiated.
    """

    def __init__(self, op: DiscreteOperator[Any], mesh: Mesh) -> None:
        self._op = op
        self._mesh = mesh

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    def __call__(self, state: State) -> State:
        """Apply the operator to a State; return a new State of residuals.

        Evaluates the stencil cell-by-cell via the symbolic DiscreteOperator.
        This path is correct for any input but carries sympy overhead; a
        Tensor-native evaluation path is the intended future replacement.
        """
        result = self._op(state)
        shape = self._mesh.shape
        n_total = self._mesh.n_cells
        values = [
            float(result(_to_multi(i, shape)))  # type: ignore[arg-type]
            for i in range(n_total)
        ]
        return State(self._mesh, Tensor(values))

    def assemble(self, backend: Any = None) -> Tensor:
        """Materialize the operator as an N×N stiffness Tensor.

        Applies the operator to each sympy unit-basis vector in turn and
        converts the resulting expressions to float.  Row and column ordering
        is axis-0-fastest.  Intended for direct or iterative linear solvers;
        not for repeated application in time integration.

        backend, when provided, places the assembled matrix on the requested
        device.
        """
        shape = self._mesh.shape
        n_total = self._mesh.n_cells
        rows: list[list[float]] = [[0.0] * n_total for _ in range(n_total)]
        for j in range(n_total):
            e_j = _BasisField(self._mesh, _to_multi(j, shape))
            lh_ej = self._op(e_j)
            for i in range(n_total):
                rows[i][j] = float(lh_ej(_to_multi(i, shape)))  # type: ignore[arg-type]
        return Tensor(rows, backend=backend)


__all__ = ["Operator"]
