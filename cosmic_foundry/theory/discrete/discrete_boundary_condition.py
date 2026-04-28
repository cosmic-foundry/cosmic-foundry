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

    def n_ghost_layers(self, order: int) -> int:
        """Ghost cell layers required on each side for a stencil of this order.

        For order p = 2n, a centered stencil accesses n cells on each side of a
        face, so the boundary cell needs n − 1 interior neighbors plus n ghost
        cells beyond the mesh edge.  The default n = order // 2 is correct for
        all standard BCs (Dirichlet, Neumann, Periodic, Inhomogeneous Dirichlet).
        Subclasses may override if their ghost-cell formula is only valid up to a
        smaller depth.
        """
        return order // 2


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


class ZeroGhostCells(DiscreteBoundaryCondition):
    """Zero-valued ghost cells (absorbing / no-BC behavior).

    Out-of-bounds indices return sympy.Integer(0); in-bounds indices
    delegate to field.  Default BC for Discretization when none is supplied.
    """

    def extend(self, field: DiscreteField[Any], mesh: Mesh) -> DiscreteField[Any]:
        shape = mesh.shape

        def extended(idx: tuple[int, ...]) -> Any:
            for i, N in zip(idx, shape, strict=True):
                if i < 0 or i >= N:
                    return sympy.Integer(0)
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


class NeumannGhostCells(DiscreteBoundaryCondition):
    """Homogeneous Neumann ghost cells via even reflection about boundary faces.

    For each axis a with mesh size N = mesh.shape[a]:
        idx[a] < 0  → reflect to −1−idx[a], keep sign   (even reflection)
        idx[a] ≥ N → reflect to 2N−1−idx[a], keep sign  (even reflection)
    Corners (out of bounds along multiple axes) are handled recursively.
    Enforces ∂φ/∂n = 0 at each boundary face (zero normal flux condition).

    Discrete counterpart of a homogeneous Neumann boundary condition.
    The assembled stiffness matrix has constant functions in its null space:
    A · ones = 0 (no net flux through a domain where all normal gradients vanish).
    """

    def extend(self, field: DiscreteField[Any], mesh: Mesh) -> DiscreteField[Any]:
        shape = mesh.shape

        def extended(idx: tuple[int, ...]) -> Any:
            for a, (i, N) in enumerate(zip(idx, shape, strict=True)):
                if i < 0:
                    reflected = idx[:a] + (-1 - i,) + idx[a + 1 :]
                    return extended(reflected)  # no sign flip: even reflection
                if i >= N:
                    reflected = idx[:a] + (2 * N - 1 - i,) + idx[a + 1 :]
                    return extended(reflected)  # no sign flip: even reflection
            return field(idx)  # type: ignore[arg-type]

        return _CallableDiscreteField(mesh, extended)


class InhomogeneousDirichletGhostCells(DiscreteBoundaryCondition):
    """Inhomogeneous Dirichlet ghost cells: φ = g at each boundary face.

    Ghost-cell formula: u_ghost = 2·g − u_mirror, where the mirror cell is the
    in-bounds cell obtained by reflecting the ghost index about the boundary face.
    This enforces the face average (u_ghost + u_mirror) / 2 = g exactly.

    g specifies the boundary value.  It may be:
      - A scalar or SymPy expression: the same value is applied to every boundary
        face of every axis.
      - A callable g(axis: int, is_low: bool) → Any: called once per ghost-cell
        lookup with the boundary axis and side (is_low=True for the low face at
        origin[axis], False for the high face).  This allows different values on
        different faces (e.g. g(0, True)=1, g(0, False)=2) without requiring
        spatial coordinates.

    Corners (out of bounds along multiple axes) are handled by applying the
    formula recursively along each out-of-bounds axis in turn.

    Reduces to DirichletGhostCells when g ≡ 0 (or g returns 0 for all calls).
    """

    def __init__(self, g: Any) -> None:
        if callable(g):
            self._g = g
        else:
            self._g = lambda _axis, _is_low: g

    def extend(self, field: DiscreteField[Any], mesh: Mesh) -> DiscreteField[Any]:
        shape = mesh.shape

        def extended(idx: tuple[int, ...]) -> Any:
            for a, (i, N) in enumerate(zip(idx, shape, strict=True)):
                if i < 0:
                    reflected = idx[:a] + (-1 - i,) + idx[a + 1 :]
                    return 2 * self._g(a, True) - extended(reflected)
                if i >= N:
                    reflected = idx[:a] + (2 * N - 1 - i,) + idx[a + 1 :]
                    return 2 * self._g(a, False) - extended(reflected)
            return field(idx)  # type: ignore[arg-type]

        return _CallableDiscreteField(mesh, extended)


__all__ = [
    "DirichletGhostCells",
    "DiscreteBoundaryCondition",
    "InhomogeneousDirichletGhostCells",
    "NeumannGhostCells",
    "PeriodicGhostCells",
    "ZeroGhostCells",
]
