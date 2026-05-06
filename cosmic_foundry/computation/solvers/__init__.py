"""Linear solver classes: ABCs and concrete algorithms."""

from cosmic_foundry.computation.solvers.capabilities import (
    LEAST_SQUARES_SOLVER_COVERAGE_REGIONS,
    LINEAR_SOLVER_COVERAGE_REGIONS,
    ROOT_SOLVER_COVERAGE_REGIONS,
    SPECTRAL_SOLVER_COVERAGE_REGIONS,
    least_squares_solver_coverage_regions,
    linear_solver_coverage_regions,
    root_solver_coverage_regions,
    select_least_squares_solver_for_descriptor,
    select_linear_solver_for_descriptor,
    select_root_solver_for_descriptor,
    select_spectral_solver_for_descriptor,
    spectral_solver_coverage_regions,
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
from cosmic_foundry.computation.solvers.newton_root_solver import (
    DirectionalDerivativeRootRelation,
    FixedPointRootRelation,
    FixedPointRootSolver,
    MatrixFreeNewtonKrylovRootSolver,
    NewtonRootSolver,
    RootRelation,
)
from cosmic_foundry.computation.solvers.relations import (
    FiniteDimensionalResidualRelation,
    LeastSquaresRelation,
    LinearResidualRelation,
)
from cosmic_foundry.computation.solvers.spectral_solver import (
    DenseSymmetricEigenpairSolver,
    SpectralSolver,
)

__all__ = [
    "DenseCGSolver",
    "DenseGMRESSolver",
    "DenseGaussSeidelSolver",
    "DenseJacobiSolver",
    "DenseLUSolver",
    "DenseSymmetricEigenpairSolver",
    "DenseSVDLeastSquaresSolver",
    "DenseSVDSolver",
    "DirectSolver",
    "DirectionalDerivativeRootRelation",
    "FixedPointRootRelation",
    "FixedPointRootSolver",
    "FiniteDimensionalResidualRelation",
    "IterativeSolver",
    "KrylovSolver",
    "LeastSquaresSolver",
    "LeastSquaresRelation",
    "LEAST_SQUARES_SOLVER_COVERAGE_REGIONS",
    "least_squares_solver_coverage_regions",
    "StationaryIterationSolver",
    "LINEAR_SOLVER_COVERAGE_REGIONS",
    "LinearOperator",
    "LinearResidualRelation",
    "linear_solver_coverage_regions",
    "LinearSolver",
    "MatrixFreeNewtonKrylovRootSolver",
    "NewtonRootSolver",
    "ROOT_SOLVER_COVERAGE_REGIONS",
    "RootRelation",
    "root_solver_coverage_regions",
    "select_least_squares_solver_for_descriptor",
    "select_linear_solver_for_descriptor",
    "select_root_solver_for_descriptor",
    "select_spectral_solver_for_descriptor",
    "SPECTRAL_SOLVER_COVERAGE_REGIONS",
    "SpectralSolver",
    "spectral_solver_coverage_regions",
]
