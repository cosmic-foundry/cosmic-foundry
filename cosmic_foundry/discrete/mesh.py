"""Mesh ABC."""

from __future__ import annotations

from abc import abstractmethod

from cosmic_foundry.continuous.manifold import Chart
from cosmic_foundry.discrete.cell_complex import CellComplex


class Mesh(CellComplex):
    """A CellComplex carrying a Chart from continuous/.

    A Mesh is a CellComplex grounded in a manifold's coordinate system
    by a Chart.  The chart's metric makes the complex geometric: faces
    are regions in the chart's parameter space, and cell volumes are
    derived from face geometry via the divergence theorem:

        |Ωᵢ| = (1/n) ∑_{f ∈ ∂Ωᵢ} xf · nf Af

    General volumes and areas are computed as ∫ √|g| dV and ∫ √|g_σ| dA
    using the metric g of the chart's domain manifold.  Volume, area, and
    normal are derived properties fully determined by the CellComplex and
    the Chart; concrete subclasses realize the derivation.

    Covers: Cartesian (g = I), cylindrical (√|g| = r),
    GR spacetimes (curved g), moving mesh (time-varying Chart).

    Required:
        chart — the chart grounding this complex in a manifold's coordinate system
    """

    @property
    @abstractmethod
    def chart(self) -> Chart:
        """The chart grounding this complex in a manifold's coordinate system."""


__all__ = ["Mesh"]
