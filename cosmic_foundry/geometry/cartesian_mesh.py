"""CartesianMesh: concrete StructuredMesh on flat Euclidean space."""

from __future__ import annotations

from itertools import combinations

import sympy

from cosmic_foundry.geometry.euclidean_manifold import EuclideanManifold
from cosmic_foundry.theory.continuous.manifold import Chart
from cosmic_foundry.theory.discrete.structured_mesh import StructuredMesh
from cosmic_foundry.theory.foundation.function import Function
from cosmic_foundry.theory.foundation.indexed_set import IndexedSet
from cosmic_foundry.theory.foundation.set import Set


class CartesianMesh(StructuredMesh):
    """A uniform Cartesian mesh on flat Euclidean space.

    CartesianMesh is the concrete simulation mesh for flat-geometry problems.
    All geometry is derived from origin, spacing, and shape under the flat
    metric g = I.  The CartesianChart is constructed internally — there is
    no free choice of chart for a Cartesian mesh.

    Free:
        origin  — coordinates of the lower corner of cell (0, …, 0)
        spacing — cell widths along each axis
        shape   — number of cells along each axis

    Derived:
        chart       — CartesianChart on EuclideanManifold(ndim)
        coordinate  — origin + (idx + ½)·spacing
        cell_volume — ∏ Δxₖ
        face_area   — ∏_{k≠j} Δxₖ for face ⊥ to axis j
        face_normal — ê_j for face ⊥ to axis j
    """

    class _CellSet(IndexedSet):
        """IndexedSet of top-dimensional cells."""

        def __init__(self, cell_shape: tuple[int, ...]) -> None:
            self._shape = cell_shape

        @property
        def shape(self) -> tuple[int, ...]:
            return self._shape

        def intersect(self, other: IndexedSet) -> IndexedSet | None:
            if not isinstance(other, CartesianMesh._CellSet):
                return None
            if self._shape != other._shape:
                return None
            return self

    class _LowerCellSet(Set):  # noqa: B024
        """Set of k-cells for k < ndim."""

        def __init__(self, count: int) -> None:
            self._count = count

        @property
        def count(self) -> int:
            """Number of k-cells in this set."""
            return self._count

    class _GeneralBoundaryMap(Function):
        """Boundary operator ∂_k for a Cartesian CW-complex.

        Implements the standard CW boundary formula for k-cells identified
        by (active_axes, idx):
          - active_axes: sorted tuple of the k axes the cell extends along
          - idx: lower-corner position in the full n-dimensional vertex grid

        For the j-th active axis aⱼ the boundary contributes:
          high face (idx + ê_{aⱼ}): sign = (−1)^j
          low  face (same idx):      sign = (−1)^{j+1}

        The output list has 2k entries
        [(remaining_axes, face_idx, sign), ...], each a (k−1)-cell in the
        same (active_axes, idx) representation.
        """

        def __init__(self, k: int) -> None:
            self._k = k

        def __call__(
            self,
            cell: tuple[tuple[int, ...], tuple[int, ...]],
        ) -> list[tuple[tuple[int, ...], tuple[int, ...], int]]:
            """Return oriented (k−1)-cell faces of *cell*.

            *cell* is (active_axes, idx).  Returns
            [(remaining_axes, face_idx, sign), ...] with 2k entries.
            """
            active_axes, idx = cell
            result: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
            for j, a in enumerate(active_axes):
                remaining = active_axes[:j] + active_axes[j + 1 :]
                sign = (-1) ** j
                hi_idx = idx[:a] + (idx[a] + 1,) + idx[a + 1 :]
                result.append((remaining, hi_idx, sign))
                result.append((remaining, idx, -sign))
            return result

    def __init__(
        self,
        origin: tuple[sympy.Expr, ...],
        spacing: tuple[sympy.Expr, ...],
        shape: tuple[int, ...],
    ) -> None:
        if len(origin) != len(spacing) or len(origin) != len(shape):
            msg = "origin, spacing, and shape must have the same length"
            raise ValueError(msg)
        if any(s < 1 for s in shape):
            msg = f"all shape entries must be >= 1, got {shape}"
            raise ValueError(msg)
        if any(s.is_positive is False for s in spacing):
            msg = f"all spacing entries must be > 0, got {spacing}"
            raise ValueError(msg)
        self._origin = origin
        self._spacing = spacing
        self._shape = shape
        ndim = len(shape)
        space = EuclideanManifold(ndim)
        self._chart: Chart = space.atlas[0]

    @property
    def chart(self) -> Chart:
        return self._chart

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __getitem__(self, k: int) -> Set:
        """Return the Set of k-cells."""
        ndim = len(self._shape)
        if k == ndim:
            return CartesianMesh._CellSet(self._shape)
        if k == 0:
            count = 1
            for s in self._shape:
                count *= s + 1
            return CartesianMesh._LowerCellSet(count)
        if 0 < k < ndim:
            return CartesianMesh._LowerCellSet(self._lower_cell_count(k))
        msg = f"k must be in [0, {ndim}], got {k}"
        raise IndexError(msg)

    def __len__(self) -> int:
        """Number of cell dimensions: ndim + 1."""
        return len(self._shape) + 1

    def boundary(self, k: int) -> Function:
        """Return the boundary operator ∂_k for k-cells (k ∈ [1, ndim]).

        The returned Function maps a k-cell identifier
        (active_axes: tuple[int,...], idx: tuple[int,...]) to a list of
        signed (k−1)-cell identifiers:
        [(remaining_axes, face_idx, sign), ...] with 2k entries.
        """
        ndim = len(self._shape)
        if k < 1 or k > ndim:
            msg = f"k must be in [1, {ndim}], got {k}"
            raise IndexError(msg)
        return CartesianMesh._GeneralBoundaryMap(k)

    def coordinate(self, idx: tuple[int, ...]) -> tuple[sympy.Expr, ...]:
        """Return cell-center coordinates: origin + (idx + ½)·spacing."""
        return tuple(
            self._origin[i]
            + (sympy.Integer(idx[i]) + sympy.Rational(1, 2)) * self._spacing[i]
            for i in range(len(self._shape))
        )

    @property
    def cell_volume(self) -> sympy.Expr:
        """Cell volume ∏ Δxₖ under flat metric."""
        result: sympy.Expr = sympy.Integer(1)
        for s in self._spacing:
            result = result * s
        return result

    def face_area(self, axis: int) -> sympy.Expr:
        """Face area ∏_{k≠axis} Δxₖ for faces perpendicular to axis."""
        ndim = len(self._shape)
        if axis < 0 or axis >= ndim:
            msg = f"axis must be in [0, {ndim - 1}], got {axis}"
            raise IndexError(msg)
        result: sympy.Expr = sympy.Integer(1)
        for i, s in enumerate(self._spacing):
            if i != axis:
                result = result * s
        return result

    def face_normal(self, axis: int) -> tuple[sympy.Integer, ...]:
        """Unit outward normal ê_axis for faces perpendicular to axis."""
        ndim = len(self._shape)
        if axis < 0 or axis >= ndim:
            msg = f"axis must be in [0, {ndim - 1}], got {axis}"
            raise IndexError(msg)
        return tuple(
            sympy.Integer(1) if i == axis else sympy.Integer(0) for i in range(ndim)
        )

    def _lower_cell_count(self, k: int) -> int:
        ndim = len(self._shape)
        total = 0
        for axes in combinations(range(ndim), k):
            axes_set = set(axes)
            count = 1
            for i in range(ndim):
                count *= self._shape[i] if i in axes_set else (self._shape[i] + 1)
            total += count
        return total


__all__ = ["CartesianMesh"]
