"""CartesianMesh: concrete StructuredMesh with flat metric."""

from __future__ import annotations

from typing import Any

import sympy

from cosmic_foundry.continuous.manifold import Chart
from cosmic_foundry.discrete.structured_mesh import StructuredMesh
from cosmic_foundry.foundation.function import Function
from cosmic_foundry.foundation.indexed_set import IndexedSet
from cosmic_foundry.foundation.set import Set


class CartesianMesh(StructuredMesh):
    """A StructuredMesh on a flat Cartesian domain with uniform spacing.

    CartesianMesh is the first concrete mesh class.  All geometry is
    derived from three free parameters — origin, spacing, and shape —
    under a flat metric (g = I).

    Free:
        origin  — coordinates of the lower corner of cell (0, …, 0)
        spacing — cell widths along each axis
        shape   — number of cells along each axis
        chart   — the Chart grounding coordinate symbols

    Derived:
        coordinate  — origin + (idx + ½)·spacing
        cell_volume — ∏ Δxₖ
        face_area   — ∏_{k≠j} Δxₖ for face ⊥ to axis j
        face_normal — ê_j for face ⊥ to axis j
    """

    class _CellSet(IndexedSet):
        """IndexedSet of top-dimensional cells in a CartesianMesh."""

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
        """Set of k-cells for k < ndim in a CartesianMesh."""

        def __init__(self, count: int) -> None:
            self._count = count

        @property
        def count(self) -> int:
            """Number of k-cells in this set."""
            return self._count

    class _BoundaryMap(Function):
        """Boundary operator ∂_k for a CartesianMesh."""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "CartesianMesh boundary map evaluation deferred to Epoch 3"
            )

    def __init__(
        self,
        origin: tuple[sympy.Expr, ...],
        spacing: tuple[sympy.Expr, ...],
        shape: tuple[int, ...],
        chart: Chart,
    ) -> None:
        if len(origin) != len(spacing) or len(origin) != len(shape):
            msg = "origin, spacing, and shape must have the same length"
            raise ValueError(msg)
        self._origin = origin
        self._spacing = spacing
        self._shape = shape
        self._chart = chart

    @property
    def chart(self) -> Chart:
        return self._chart

    def __getitem__(self, k: int) -> Set:
        """Return the Set of k-cells.

        For k == ndim, returns an IndexedSet with the cell shape.
        For k < ndim, returns a Set with the appropriate cell count.
        """
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
        """Return the boundary operator ∂_k.

        The boundary map is structurally defined but evaluation is
        deferred to Epoch 3 when DiscreteOperator is implemented.
        """
        if k <= 0 or k > len(self._shape):
            msg = f"k must be in [1, {len(self._shape)}], got {k}"
            raise IndexError(msg)
        return CartesianMesh._BoundaryMap()

    def coordinate(self, idx: tuple[int, ...]) -> tuple[sympy.Expr, ...]:
        """Return cell-center coordinates: origin + (idx + ½)·spacing."""
        return tuple(
            self._origin[i]
            + (sympy.Integer(idx[i]) + sympy.Rational(1, 2)) * self._spacing[i]
            for i in range(len(self._shape))
        )

    @property
    def cell_volume(self) -> sympy.Expr:
        """Cell volume ∏ Δxₖ; derived from spacing under flat metric."""
        result: sympy.Expr = sympy.Integer(1)
        for s in self._spacing:
            result = result * s
        return result

    def face_area(self, axis: int) -> sympy.Expr:
        """Face area ∏_{k≠axis} Δxₖ for faces perpendicular to the given axis."""
        result: sympy.Expr = sympy.Integer(1)
        for i, s in enumerate(self._spacing):
            if i != axis:
                result = result * s
        return result

    def face_normal(self, axis: int) -> tuple[sympy.Integer, ...]:
        """Unit face normal ê_axis for faces perpendicular to the given axis."""
        return tuple(
            sympy.Integer(1) if i == axis else sympy.Integer(0)
            for i in range(len(self._shape))
        )

    def _lower_cell_count(self, k: int) -> int:
        """Count k-cells in a Cartesian grid.

        For a grid with shape (s₁, …, sₙ), the number of k-cells is the
        sum over all k-element subsets of axes: for each subset, the axes
        in the subset contribute sᵢ cells and the remaining axes contribute
        (sᵢ + 1) cells.
        """
        from itertools import combinations

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
