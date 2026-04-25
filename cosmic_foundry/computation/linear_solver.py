"""LinearSolver ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cosmic_foundry.theory.discrete.discretization import Discretization
from cosmic_foundry.theory.discrete.mesh_function import MeshFunction


class LinearSolver(ABC):
    """Abstract interface for solving the linear discrete equation Lₕ u = f.

    A LinearSolver takes a Discretization (which produces the linear operator
    Lₕ via __call__ and assembles the stiffness matrix via assemble_matrix)
    together with a right-hand-side MeshFunction f, and returns a MeshFunction
    u satisfying Lₕ u = f.  The solver is scoped to linear operators: those
    whose assembled matrix A is a well-defined linear map from cell averages
    to cell residuals.  Nonlinear operators (HyperbolicFlux, Euler equations)
    require a separate NonlinearSolver hierarchy.

    In plain terms: given a discrete PDE assembled into a matrix system Au = f,
    find u.  The Discretization owns the assembly; the LinearSolver owns only
    the iteration strategy.  This separation lets solvers be swapped without
    touching discretization code.

    Required:
        solve — apply the iteration and return the solution MeshFunction
    """

    @abstractmethod
    def solve(
        self,
        discretization: Discretization,
        rhs: MeshFunction,
    ) -> MeshFunction:
        """Solve Lₕ u = rhs for u; return the solution MeshFunction.

        The solution is well-defined only when the operator assembled by
        discretization is invertible (e.g. the SPD operator from
        FVMDiscretization with DirichletBC, proved in C6).  No convergence
        guarantee is made for non-invertible or indefinite operators.

        Parameters
        ----------
        discretization:
            Supplies Lₕ via assemble_matrix() and mesh geometry.
        rhs:
            The right-hand-side MeshFunction f; must be callable with cell
            multi-indices and return values convertible to float.

        Returns
        -------
        MeshFunction
            u such that ‖Lₕ u − f‖ < solver tolerance (implementation-defined
            norm and convergence criterion).
        """


__all__ = ["LinearSolver"]
