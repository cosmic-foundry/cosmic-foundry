"""StructuredMesh ABC."""

from __future__ import annotations

from abc import abstractmethod

import sympy

from cosmic_foundry.continuous.field import Field
from cosmic_foundry.discrete.mesh import Mesh


class StructuredMesh(Mesh):
    """A Mesh whose cells are regular and axis-aligned.

    A StructuredMesh carries a Chart grounding coordinate symbols
    symbolically and adds the abstract coordinate function mapping
    multi-indices to chart coordinates.  The evaluation bridge

        field.expr.subs(zip(chart.symbols, coordinate(idx)))

    connects continuous field expressions to discrete mesh values.

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
        """Evaluate a field at cell center idx via coordinate substitution.

        Derived from coordinate and chart.symbols: fully determined by the
        StructuredMesh and the Field's symbolic expression.
        """
        pairs = list(zip(self.chart.symbols, self.coordinate(idx), strict=False))
        return field.expr.subs(pairs)


__all__ = ["StructuredMesh"]
