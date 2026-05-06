"""Spectral solver interface and dense symmetric eigenpair implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cosmic_foundry.computation.tensor import Tensor


class SpectralSolver(ABC):
    """Abstract interface for matrix spectral solves."""

    @abstractmethod
    def solve(self, matrix: Tensor) -> tuple[float, Tensor]:
        """Return an eigenvalue/eigenvector pair for an assembled matrix."""


class DenseSymmetricEigenpairSolver(SpectralSolver):
    """Dense symmetric eigenpair solver backed by Hermitian eigendecomposition."""

    def solve(self, matrix: Tensor) -> tuple[float, Tensor]:
        """Return the dominant eigenpair for a real symmetric matrix."""
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("dense symmetric eigenpair solve requires a square matrix")

        array = np.asarray(matrix.to_list(), dtype=float)
        if not np.allclose(array, array.T, rtol=1e-12, atol=1e-12):
            raise ValueError("dense symmetric eigenpair solve requires symmetry")

        eigenvalues, eigenvectors = np.linalg.eigh(array)
        index = int(np.argmax(eigenvalues))
        eigenvalue = float(eigenvalues[index])
        eigenvector = eigenvectors[:, index]
        pivot = int(np.argmax(np.abs(eigenvector)))
        if eigenvector[pivot] < 0.0:
            eigenvector = -eigenvector
        return eigenvalue, Tensor(eigenvector.tolist(), backend=matrix.backend)


__all__ = ["DenseSymmetricEigenpairSolver", "SpectralSolver"]
