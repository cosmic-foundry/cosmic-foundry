"""Newton solver for finite-dimensional root relations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cosmic_foundry.computation.algorithm_capabilities import StructuredPredicate
from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.solvers.coverage import nonlinear_root_predicates
from cosmic_foundry.computation.tensor import Tensor, einsum, norm


@dataclass(frozen=True)
class RootSolveProblem:
    """Finite-dimensional residual relation ``F(x) = 0``."""

    residual: Callable[[Tensor], Tensor]
    jacobian: Callable[[Tensor], Tensor]
    initial: Tensor
    equality_constraint_gradients: Tensor | None = None

    @property
    def equality_constraint_count(self) -> int:
        """Return the number of equality constraints active in the relation."""
        if self.equality_constraint_gradients is None:
            return 0
        return self.equality_constraint_gradients.shape[0]


class NewtonRootSolver:
    """Solve ``F(x) = 0`` by Newton iteration."""

    root_solver_coverage: tuple[tuple[StructuredPredicate, ...], ...] = (
        nonlinear_root_predicates()
    )

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        tolerance: float = 1e-12,
    ) -> None:
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._lu = LUFactorization()

    def solve(
        self,
        problem: RootSolveProblem,
    ) -> Tensor:
        """Return a root of the supplied residual relation."""
        x = problem.initial
        gram: Tensor | None = None
        if problem.equality_constraint_gradients is not None:
            gradients = problem.equality_constraint_gradients
            gram = einsum("ij,kj->ik", gradients, gradients)
        for _ in range(self._max_iterations):
            Fx = problem.residual(x)
            if self._small(Fx, x):
                break
            delta = self._lu.factorize(problem.jacobian(x)).solve(
                Tensor.zeros(x.shape[0], backend=x.backend) - Fx
            )
            if problem.equality_constraint_gradients is not None and gram is not None:
                gradients = problem.equality_constraint_gradients
                xi = self._lu.factorize(gram).solve(gradients @ delta)
                delta = delta - einsum("ij,i->j", gradients, xi)
            x = x + delta
            if self._small(delta, x):
                break
        return x

    def _small(self, vector: Tensor, scale: Tensor) -> bool:
        return float(norm(vector)) < self._tolerance * (1.0 + float(norm(scale)))


__all__ = ["NewtonRootSolver", "RootSolveProblem"]
