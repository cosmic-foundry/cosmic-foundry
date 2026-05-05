"""Newton solver for finite-dimensional root relations."""

from __future__ import annotations

from collections.abc import Callable

from cosmic_foundry.computation.algorithm_capabilities import StructuredPredicate
from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.solvers.coverage import nonlinear_root_predicates
from cosmic_foundry.computation.tensor import Tensor, einsum, norm


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
        residual: Callable[[Tensor], Tensor],
        jacobian: Callable[[Tensor], Tensor],
        initial: Tensor,
        *,
        constraint_gradients: Tensor | None = None,
    ) -> Tensor:
        """Return a root, optionally projecting Newton steps into ``null(C)``."""
        x = initial
        gram: Tensor | None = None
        if constraint_gradients is not None:
            gram = einsum("ij,kj->ik", constraint_gradients, constraint_gradients)
        for _ in range(self._max_iterations):
            Fx = residual(x)
            if self._small(Fx, x):
                break
            delta = self._lu.factorize(jacobian(x)).solve(
                Tensor.zeros(x.shape[0], backend=x.backend) - Fx
            )
            if constraint_gradients is not None and gram is not None:
                xi = self._lu.factorize(gram).solve(constraint_gradients @ delta)
                delta = delta - einsum("ij,i->j", constraint_gradients, xi)
            x = x + delta
            if self._small(delta, x):
                break
        return x

    def _small(self, vector: Tensor, scale: Tensor) -> bool:
        return float(norm(vector)) < self._tolerance * (1.0 + float(norm(scale)))


__all__ = ["NewtonRootSolver"]
