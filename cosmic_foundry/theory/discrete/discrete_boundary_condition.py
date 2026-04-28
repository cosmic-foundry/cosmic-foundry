"""DiscreteBoundaryCondition: ghost-cell extension rules for discrete operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import sympy

from cosmic_foundry.theory.discrete.discrete_field import (
    DiscreteField,
    _CallableDiscreteField,
)
from cosmic_foundry.theory.discrete.mesh import Mesh


def _apply_zero_ghosts(field: DiscreteField[Any], mesh: Mesh) -> DiscreteField[Any]:
    """Extend field with zero-valued ghost cells (absorbing/no-BC behavior).

    Out-of-bounds indices return sympy.Integer(0); in-bounds indices
    delegate to field.  Used by FVMDiscretization and FDDiscretization
    when no DiscreteBoundaryCondition is supplied.
    """
    shape = mesh.shape

    def extended(idx: tuple[int, ...]) -> Any:
        for i, N in zip(idx, shape, strict=True):
            if i < 0 or i >= N:
                return sympy.Integer(0)
        return field(idx)  # type: ignore[arg-type]

    return _CallableDiscreteField(mesh, extended)


class DiscreteBoundaryCondition(ABC):
    """Abstract ghost-cell extension rule for FVM operators.

    DiscreteBoundaryCondition is the discrete counterpart of BoundaryCondition:
    while the continuous BC describes the mathematical constraint (φ|_∂Ω = g),
    the discrete BC describes how to extend a field beyond the mesh boundary so
    that NumericalFlux stencils can be evaluated at boundary-adjacent cells.

    Each concrete subclass owns its own ghost-cell rule and returns a new
    DiscreteField defined on all integer indices, not just the in-bounds subset.
    FVM operators call extend() once before evaluating face fluxes, eliminating
    isinstance dispatch from the operator itself.
    """

    @abstractmethod
    def extend(self, field: DiscreteField[Any], mesh: Mesh) -> DiscreteField[Any]:
        """Return a ghost-cell-extended version of field.

        Parameters
        ----------
        field:
            DiscreteField defined for in-bounds cell indices
            (0 ≤ idx[a] < mesh.shape[a] for each axis a).
        mesh:
            The mesh whose shape determines the in-bounds region.

        Returns
        -------
        DiscreteField that agrees with field for in-bounds indices and returns
        ghost values for out-of-bounds indices consistent with this BC.
        """


class DirichletGhostCells(DiscreteBoundaryCondition):
    """Homogeneous Dirichlet ghost cells via odd reflection about boundary faces.

    For each axis a with mesh size N = mesh.shape[a]:
        idx[a] < 0  → reflect to −1−idx[a], negate value
        idx[a] ≥ N → reflect to 2N−1−idx[a], negate value
    Corners (out of bounds along multiple axes) are handled recursively.
    Enforces φ = 0 at each boundary face by bilinear interpolation.

    Discrete counterpart of DirichletBC.
    """

    def extend(self, field: DiscreteField[Any], mesh: Mesh) -> DiscreteField[Any]:
        shape = mesh.shape

        def extended(idx: tuple[int, ...]) -> Any:
            for a, (i, N) in enumerate(zip(idx, shape, strict=True)):
                if i < 0:
                    reflected = idx[:a] + (-1 - i,) + idx[a + 1 :]
                    return -extended(reflected)
                if i >= N:
                    reflected = idx[:a] + (2 * N - 1 - i,) + idx[a + 1 :]
                    return -extended(reflected)
            return field(idx)  # type: ignore[arg-type]

        return _CallableDiscreteField(mesh, extended)


class PeriodicGhostCells(DiscreteBoundaryCondition):
    """Periodic ghost cells via wrap-around indexing modulo mesh shape.

    For each axis a with mesh size N = mesh.shape[a]:
        idx[a] mod N  (handles both negative and out-of-bounds positive indices)

    Discrete counterpart of PeriodicBC.
    """

    def extend(self, field: DiscreteField[Any], mesh: Mesh) -> DiscreteField[Any]:
        shape = mesh.shape

        def extended(idx: tuple[int, ...]) -> Any:
            wrapped = tuple(i % N for i, N in zip(idx, shape, strict=True))
            return field(wrapped)  # type: ignore[arg-type]

        return _CallableDiscreteField(mesh, extended)


__all__ = ["DiscreteBoundaryCondition", "DirichletGhostCells", "PeriodicGhostCells"]
