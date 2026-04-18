"""Field hierarchy.

- ``Field``         — abstract base: f: M → V; inherits Function.
- ``ScalarField``   — marker for V = ℝ.
- ``TensorField``   — abstract for V = T^(p,q)M; carries ``tensor_type``.
- ``ContinuousField``— concrete scalar field stored as an analytic callable.
- ``PatchFunction`` — concrete scalar field stored as a JAX array payload.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cosmic_foundry.function import Function


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

    def sample(self, *coordinate_arrays: Any) -> PatchFunction:
        """Sample f at the given coordinate arrays, returning a PatchFunction."""
        return PatchFunction(name=self.name, payload=self.execute(*coordinate_arrays))


@dataclass(frozen=True)
class PatchFunction(ScalarField):
    """A discrete scalar field stored as a JAX array payload.

    The concrete data-side counterpart of ``Patch`` (the geometry).
    ``payload`` holds the field values at the DOF locations defined by the
    associated Patch; spatial metadata lives in the Patch, not here.
    """

    name: str
    payload: Any

    def execute(self) -> Any:
        """Return the stored payload array."""
        return self.payload


__all__ = [
    "ContinuousField",
    "Field",
    "PatchFunction",
    "ScalarField",
    "TensorField",
]
