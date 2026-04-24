"""StructuredMesh ABC."""

from __future__ import annotations

from abc import abstractmethod

import sympy

from cosmic_foundry.theory.continuous.field import Field
from cosmic_foundry.theory.continuous.manifold import Point
from cosmic_foundry.theory.discrete.mesh import Mesh


class StructuredMesh(Mesh):
    """A Mesh whose cells are regular and axis-aligned.

    A StructuredMesh carries a Chart grounding coordinate symbols
    symbolically and adds the abstract coordinate function mapping
    multi-indices to chart coordinates.  The evaluation bridge —
    field(Point(manifold, chart, coordinate(idx))) — connects continuous
    field expressions to discrete mesh values via the typed Point interface.

    The regularity constraint narrows the top-dimensional cell set
    complex[n] from Set to IndexedSet: the cells biject with a
    rectangular region of ℤⁿ, earning shape, ndim, and intersect
    as derived properties of cell regularity.

    Required:
        coordinate — return the chart coordinates of the cell center at idx

    Derived:
        evaluate   — evaluate a field at a cell center via coordinate substitution
    """

    @abstractmethod
    def coordinate(self, idx: tuple[int, ...]) -> tuple[sympy.Expr, ...]:
        """Return the chart coordinates of the cell center at multi-index idx."""

    def evaluate(self, field: Field, idx: tuple[int, ...]) -> sympy.Expr:
        """Evaluate a field at cell center idx via the typed Point interface.

        Constructs a Point using this mesh's chart and chart.domain as the
        manifold — not field.manifold — to preserve the Point invariant that
        chart: Chart[M, Any] and manifold: M refer to the same M.  Field
        compatibility is enforced by the chart-symbol check in
        SymbolicFunction.__call__: if the field's symbols differ from this
        chart's symbols, a ValueError is raised before any substitution occurs.
        """
        coords = self.coordinate(idx)
        point = Point(manifold=self.chart.domain, chart=self.chart, coords=coords)
        return field(point)


__all__ = ["StructuredMesh"]
