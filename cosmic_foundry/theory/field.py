"""Field hierarchy.

- ``Field``          — abstract base: f: M → V; inherits Function.
- ``ScalarField``    — marker for V = ℝ.
- ``TensorField``    — abstract for V = T^(p,q)M; carries ``tensor_type``.
- ``ContinuousField``— concrete scalar field stored as an analytic callable.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cosmic_foundry.theory.function import Function


class Field(Function):
    """Abstract base for all fields: f: M → V.

    A field assigns a value in V to every point in a manifold M.
    Subclasses specialize by the codomain V (scalar, tensor) and by
    how f is represented (analytic callable, discrete array, modal
    coefficients).

    ``name`` is required on all concrete subclasses (carried as a
    frozen-dataclass field) and identifies the physical quantity.
    """

    name: str


class ScalarField(Field):  # noqa: B024
    """A field with codomain V = ℝ.

    Marker subclass; ``execute`` remains abstract from ``Function``.
    """


class TensorField(Field):
    """A field with codomain V = T^(p,q)M.

    Subclasses must declare the tensor type (p contravariant, q covariant
    indices).
    """

    @property
    @abstractmethod
    def tensor_type(self) -> tuple[int, int]:
        """Return (p, q) where p is contravariant and q is covariant rank."""


@dataclass(frozen=True)
class ContinuousField(ScalarField):
    """A continuous scalar field f: M → ℝ represented by an analytic callable.

    Exact representation — the callable *is* the field, not an approximation.
    Evaluated at a point by calling fn(*coords) where each coord is a JAX
    array of positions along one axis of M.
    """

    name: str
    fn: Callable[..., Any]

    def execute(self, *args: Any) -> Any:
        """Evaluate the field at the given coordinates."""
        import jax.numpy as jnp

        return jnp.asarray(self.fn(*args), dtype=jnp.float64)

    def discretize(self, mesh: Any) -> Any:
        """Sample f at each patch's node positions, returning Array[T].

        T is the backend array type.  The returned Array has the same
        Placement as the mesh: element i is the array of field values on
        patch i, with shape equal to patch.index_extent.shape and no ghost
        cells.
        """
        import jax.numpy as jnp

        from cosmic_foundry.record import Array

        elements: list[Any] = []
        for patch in mesh.elements:
            axes = [patch.node_positions(axis) for axis in range(patch.ndim)]
            coords = jnp.meshgrid(*axes, indexing="ij")
            elements.append(self.execute(*coords))
        return Array(elements=tuple(elements), placement=mesh.placement)


__all__ = [
    "ContinuousField",
    "Field",
    "ScalarField",
    "TensorField",
]
