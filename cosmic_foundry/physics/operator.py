"""Operator: physics-layer discrete operator instantiated on a Mesh."""

from __future__ import annotations

from typing import Any

from cosmic_foundry.computation.backends import get_default_backend
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.physics.state import State
from cosmic_foundry.theory.discrete.discrete_operator import DiscreteOperator
from cosmic_foundry.theory.discrete.mesh import Mesh


def _to_multi(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    idx = []
    for n in shape:
        idx.append(flat % n)
        flat //= n
    return tuple(idx)


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
        """Materialize the operator as an N×N stiffness Tensor via scatter.

        Reads precomputed stiffness_values, row_indices, col_indices from the
        DiscreteOperator and scatters them into a zero matrix in one pass.
        Row and column ordering is axis-0-fastest.  Intended for direct or
        iterative linear solvers; not for repeated application in time integration.
        """
        b = backend if backend is not None else get_default_backend()
        n = self._mesh.n_cells
        dst = b.zeros((n, n))
        vals = b.to_native(self._op.stiffness_values)
        raw = b.scatter_add(dst, self._op.row_indices, self._op.col_indices, vals)
        return Tensor(raw, backend=b)


__all__ = ["Operator"]
