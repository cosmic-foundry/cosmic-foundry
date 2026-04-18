"""Field hierarchy.

- ``Field``          — abstract base for all field parameterizations: f: D → ℝ.
- ``ContinuousField``— Θ = ∅: f: D → ℝ represented by an analytic callable.
- ``DiscreteField``  — Θ = {h}: named array payload; pure mathematical concept
                       with no spatial metadata.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
        """Evaluate the field at the given coordinates in domain D."""
        import jax.numpy as jnp

        return jnp.asarray(self.fn(*args), dtype=jnp.float64)

    def sample(self, *coordinate_arrays: Any) -> DiscreteField:
        """Sample f at the given coordinate arrays, returning a DiscreteField.

        Each argument is a JAX array of coordinates along one axis of D.
        Returns a ``DiscreteField`` with ``name`` from this field and
        ``payload = f(coordinate_arrays...)``.  No spatial metadata is attached.
        """
        return DiscreteField(name=self.name, payload=self.evaluate(*coordinate_arrays))


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


__all__ = [
    "ContinuousField",
    "DiscreteField",
    "Field",
    "Placement",
]
