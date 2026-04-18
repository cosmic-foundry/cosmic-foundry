"""Field hierarchy and FieldDiscretization.

- ``Field``              — abstract base for all field parameterizations: f: D → ℝ.
- ``ContinuousField``    — Θ = ∅: f: D → ℝ represented by an analytic callable.
- ``DiscreteField``      — Θ = {h}: named array payload; pure mathematical concept
                           with no spatial metadata.
- ``FieldDiscretization``— map from ContinuousField × UniformGrid to DistributedField
                           (spatial-domain implementation; concept is domain-general).
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cosmic_foundry.kernels import ComponentId, Map
from cosmic_foundry.mesh import DistributedField, FieldSegment
from cosmic_foundry.record import Placement


class Field(ABC):
    """Abstract base for all field parameterizations: f: D → ℝ.

    A field assigns a value to every point in its domain D. Concrete
    subclasses differ in how D is represented and how f is stored:

    - ``ContinuousField``: D = any domain, Θ = ∅, stored as a callable.
    - ``DiscreteField``:   D = D_h ⊂ D,   Θ = {h}, stored as a named array.
    """

    name: str


@dataclass(frozen=True)
class ContinuousField(Field):
    """A continuous scalar field f: D → ℝ represented by an analytic callable.

    Θ = ∅ — exact representation; the callable is the field itself, not an
    approximation of it.  D may be any domain: physical space, thermodynamic
    state space, or otherwise.  Evaluated at a point in D by calling
    fn(*args) where each arg is a JAX array of coordinates along one axis
    of D.
    """

    name: str
    fn: Callable[..., Any]

    def evaluate(self, *args: Any) -> Any:
        """Evaluate the field at the given point in domain D."""
        import jax.numpy as jnp

        return jnp.asarray(self.fn(*args), dtype=jnp.float64)


@dataclass(frozen=True)
class DiscreteField(Field):
    """A discrete scalar field f_h: Ω_h → ℝ. Θ = {h}.

    Pure mathematical concept: a named array payload with no spatial metadata.
    Spatial location, block identity, and ownership are carried by
    ``FieldSegment`` and ``DistributedField`` in the mesh layer.

    Approximation error is O(h^p) for smooth fields; p depends on the
    discretization scheme that produced this field.
    """

    name: str
    payload: Any


@dataclass(frozen=True)
class FieldDiscretization(Map):
    """Discretize a ContinuousField onto a discrete grid of points in its domain.

    The concept is domain-general: sampling f: D → ℝ onto D_h ⊂ D is the
    same operation whether D is physical space or thermodynamic state space.
    This implementation covers the spatial case (D = Ω ⊆ ℝⁿ, G = UniformGrid).

    Map:
        domain   — (f: ContinuousField on Ω ⊆ ℝⁿ, G = {(B_i, h)}) — a
                   continuous scalar field and a uniform grid partitioning Ω
                   into blocks B_i with grid spacing h; f.evaluate is called
                   with one JAX coordinate array per spatial axis,
                   broadcast-compatible with block shape
        codomain — f_h: DistributedField on Ω_h — one FieldSegment per block
                   with extent = block.index_extent and no ghost cells;
                   collected into a DistributedField over the full grid
        operator — (f, G) ↦ f_h where f_h(x_i) = f(x_i) for x_i ∈ Ω_h^int

    Θ = {h}, p = 1 — piecewise-constant representation has L∞ error O(h)
    for smooth f; verified by MMS.
    """

    def execute(self, f: ContinuousField, grid: Any) -> DistributedField:
        """Return a DistributedField with payloads equal to f at cell centers."""
        import jax.numpy as jnp

        leaves: list[FieldSegment] = []
        owners: dict[ComponentId, int] = {}
        for block in grid.blocks:
            axes = [block.cell_centers(axis) for axis in range(block.ndim)]
            coords = jnp.meshgrid(*axes, indexing="ij")
            payload = jnp.asarray(f.evaluate(*coords), dtype=jnp.float64)
            leaves.append(
                FieldSegment(
                    name=f.name,
                    segment_id=block.block_id,
                    payload=payload,
                    extent=block.index_extent,
                )
            )
            owners[block.block_id] = grid.owner(block.block_id)

        return DistributedField(
            name=f.name,
            segments=tuple(leaves),
            placement=Placement(owners),
        )


__all__ = [
    "ContinuousField",
    "DiscreteField",
    "Field",
    "FieldDiscretization",
    "Placement",
]
