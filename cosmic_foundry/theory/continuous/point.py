"""Point[M]: a manifold point expressed as coordinates in a specific chart."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from cosmic_foundry.theory.continuous.manifold import Chart

M = TypeVar("M")


@dataclass(frozen=True)
class Point(Generic[M]):
    """A point on manifold M expressed as coordinates in a specific chart.

    A manifold point is chart-independent mathematically, but to compute with
    it you must choose a chart and record which one.  Point carries both so
    that Field.evaluate can verify the chart matches the field's symbols and
    perform the correct substitution.

    Required:
        manifold — the manifold this point belongs to
        chart    — the chart whose coordinate system the coords are expressed in
        coords   — coordinate values in the order defined by chart.symbols
    """

    manifold: M
    chart: Chart[M, Any]
    coords: tuple[Any, ...]


__all__ = ["Point"]
