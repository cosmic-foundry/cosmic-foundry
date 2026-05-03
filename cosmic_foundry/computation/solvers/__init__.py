"""Linear solver classes: ABCs and concrete algorithms."""

from cosmic_foundry.computation.solvers.capabilities import (
    LINEAR_SOLVER_COVERAGE_REGIONS,
    linear_solver_coverage_regions,
    select_linear_solver_for_descriptor,
)
from cosmic_foundry.computation.solvers.dense_cg_solver import DenseCGSolver
from cosmic_foundry.computation.solvers.dense_gauss_seidel_solver import (
    DenseGaussSeidelSolver,
)
from cosmic_foundry.computation.solvers.dense_gmres_solver import DenseGMRESSolver
from cosmic_foundry.computation.solvers.dense_jacobi_solver import DenseJacobiSolver
from cosmic_foundry.computation.solvers.dense_lu_solver import DenseLUSolver
from cosmic_foundry.computation.solvers.dense_svd_solver import DenseSVDSolver
from cosmic_foundry.computation.solvers.direct_solver import DirectSolver
from cosmic_foundry.computation.solvers.iterative_solver import (
    IterativeSolver,
    KrylovSolver,
    StationaryIterationSolver,
)
from cosmic_foundry.computation.solvers.least_squares_solver import (
    DenseSVDLeastSquaresSolver,
    LeastSquaresSolver,
)
from cosmic_foundry.computation.solvers.linear_solver import (
    LinearOperator,
    LinearSolver,
)

__all__ = [
    "DenseCGSolver",
    "DenseGMRESSolver",
    "DenseGaussSeidelSolver",
    "DenseJacobiSolver",
    "DenseLUSolver",
    "DenseSVDLeastSquaresSolver",
    "DenseSVDSolver",
    "DirectSolver",
    "IterativeSolver",
    "KrylovSolver",
    "LeastSquaresSolver",
    "StationaryIterationSolver",
    "LINEAR_SOLVER_COVERAGE_REGIONS",
    "LinearOperator",
    "linear_solver_coverage_regions",
    "LinearSolver",
    "select_linear_solver_for_descriptor",
]
