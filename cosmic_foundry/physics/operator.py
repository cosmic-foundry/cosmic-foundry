"""Operator: physics-layer Tensor-backed linear operator on State objects."""

from __future__ import annotations

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.physics.state import State
from cosmic_foundry.theory.discrete.mesh import Mesh


class Operator:
    """Physics-layer Tensor-backed linear operator.

    The materialized form of a discrete linear map: a mesh-bound N×N Tensor
    produced by a discretization (e.g. FVMDiscretization.to_operator()) and
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


__all__ = ["Operator"]
