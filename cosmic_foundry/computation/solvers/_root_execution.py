"""Execution helpers for finite-dimensional root relations."""

from __future__ import annotations

from typing import Protocol, cast

from cosmic_foundry.computation.solvers.capabilities import (
    select_root_solver_for_descriptor,
)
from cosmic_foundry.computation.solvers.newton_root_solver import (
    DirectionalDerivativeRootRelation,
    FixedPointRootRelation,
    RootRelation,
)
from cosmic_foundry.computation.tensor import Tensor


class _RootSolverProtocol(Protocol):
    def solve(
        self,
        relation: (
            RootRelation | DirectionalDerivativeRootRelation | FixedPointRootRelation
        ),
    ) -> Tensor: ...


def solve_root_relation(
    relation: RootRelation | DirectionalDerivativeRootRelation | FixedPointRootRelation,
) -> Tensor:
    """Solve a root relation by querying primitive solve-relation coverage."""
    solver_type = select_root_solver_for_descriptor(
        relation.solve_relation_descriptor()
    )
    solver = cast(_RootSolverProtocol, solver_type())
    return solver.solve(relation)
