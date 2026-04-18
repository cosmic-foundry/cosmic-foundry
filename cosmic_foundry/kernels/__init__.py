"""Kernel interface primitives."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.domain import Domain
from cosmic_foundry.map import Map
from cosmic_foundry.record import ComponentId, Record
from cosmic_foundry.sink import Sink
from cosmic_foundry.source import Source


class Descriptor(ABC):
    """Abstract base for all computation-configuring value objects.

    A Descriptor is an immutable object that specifies *how* computation is
    performed — coordinate extents, iteration regions, access patterns, field
    bindings, data placement. Descriptors are distinct from Records (which are
    *about* the simulation) and from Maps (which *perform* computation).

    Every Descriptor must be serializable to a plain dict.
    """

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this descriptor."""


@dataclass(frozen=True)
class AccessPattern(Descriptor):
    """Concrete descriptor specifying the locality contract for an Op.

    ``radii[i]`` is the ghost-cell depth required on axis ``i``.
    """

    radii: tuple[int, ...]

    def halo_width(self, axis: int) -> int:
        """Return the ghost-cell depth required on one axis."""
        return self.radii[axis]

    @classmethod
    def seven_point(cls) -> AccessPattern:
        """Return the 3-D seven-point nearest-neighbor access pattern."""
        return cls((1, 1, 1))

    @classmethod
    def symmetric(cls, order: int, ndim: int = 3) -> AccessPattern:
        """Return a symmetric access pattern for an even finite-difference order."""
        if order <= 0 or order % 2 != 0:
            msg = "symmetric order must be a positive even integer"
            raise ValueError(msg)
        if ndim <= 0:
            msg = "ndim must be positive"
            raise ValueError(msg)
        return cls((order // 2,) * ndim)

    def as_dict(self) -> dict[str, Any]:
        return {"type": "AccessPattern", "radii": list(self.radii)}


@dataclass(frozen=True)
class Extent(Descriptor):
    """Half-open integer index extent."""

    slices: tuple[slice, ...]

    def __hash__(self) -> int:
        # slice objects are not hashable; derive the hash from their bounds.
        return hash(tuple((s.start, s.stop) for s in self.slices))

    @classmethod
    def from_shape(cls, shape: Sequence[int]) -> Extent:
        """Create an extent covering a full array shape."""
        return cls(tuple(slice(0, int(n)) for n in shape))

    @property
    def ndim(self) -> int:
        """Number of axes in the extent."""
        return len(self.slices)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape implied by the extent."""
        return tuple(_slice_length(axis_slice) for axis_slice in self.slices)

    def expand(self, access_pattern: AccessPattern) -> Extent:
        """Return the extent plus the halo required by an access pattern."""
        expanded = []
        for axis, axis_slice in enumerate(self.slices):
            start, stop = _checked_bounds(axis_slice)
            halo = access_pattern.halo_width(axis)
            expanded.append(slice(start - halo, stop + halo))
        return Extent(tuple(expanded))

    def as_dict(self) -> dict[str, Any]:
        return {"slices": [(s.start, s.stop) for s in self.slices]}


@dataclass(frozen=True)
class Region(Descriptor):
    """Iteration coordinates for a Map execution.

    ``n_blocks`` signals a batched region: inputs are expected to carry a
    leading batch axis of that size, and ``execute_pointwise`` lowers the
    kernel with ``jax.vmap`` so the Map remains unaware of the batch dimension.
    """

    extent: Extent
    n_blocks: int = 1

    def __post_init__(self) -> None:
        if self.n_blocks < 1:
            msg = "Region.n_blocks must be a positive integer"
            raise ValueError(msg)

    def as_dict(self) -> dict[str, Any]:
        return {"extent": self.extent.as_dict(), "n_blocks": self.n_blocks}


def execute_pointwise(
    map_like: Any,
    region: Region,
    *field_arrays: Any,
) -> Any:
    """Apply map_like._fn over region with JAX JIT and input validation.

    ``map_like`` must be hashable (for JIT caching) and expose:

    - ``_fn(*field_arrays, *index_meshgrids) -> scalar``
    - ``access_pattern: AccessPattern``

    When ``region.n_blocks > 1`` the kernel is lifted with ``jax.vmap``
    so ``_fn`` remains unaware of the batch dimension.
    """
    _validate_region_access(region, map_like.access_pattern, field_arrays)
    return _make_jit_kernel(cast(Any, map_like), region)(*field_arrays)


@functools.lru_cache(maxsize=256)
def _make_jit_kernel(map_like: Any, region: Any) -> Callable[..., Any]:
    """Return a cached JIT-compiled kernel for *(map_like, region)*.

    Both arguments must be hashable.  Frozen dataclasses satisfy this via
    their auto-generated ``__hash__``.  ``Region`` satisfies it as a frozen
    dataclass (with ``Extent.__hash__`` converting slice bounds to a tuple).

    Caching ensures repeated ``execute_pointwise`` calls with the same
    map_like and Region reuse the same compiled XLA computation.
    """
    if region.n_blocks > 1:

        @jax.jit
        def apply_batched(*jit_inputs: Any) -> Any:
            indices = _region_indices(region)

            def single_block(*block_inputs: Any) -> Any:
                return map_like._fn(*block_inputs, *indices)

            return jax.vmap(single_block)(*jit_inputs)

        return cast(Callable[..., Any], apply_batched)

    @jax.jit
    def apply(*jit_inputs: Any) -> Any:
        indices = _region_indices(region)
        return map_like._fn(*jit_inputs, *indices)

    return cast(Callable[..., Any], apply)


def _region_indices(region: Region) -> tuple[jax.Array, ...]:
    axes = []
    for axis_slice in region.extent.slices:
        start, stop = _checked_bounds(axis_slice)
        axes.append(jnp.arange(start, stop))
    return tuple(jnp.meshgrid(*axes, indexing="ij"))


def _validate_region_access(
    region: Region,
    access_pattern: AccessPattern,
    inputs: tuple[Any, ...],
) -> None:
    required = region.extent.expand(access_pattern)
    batched = region.n_blocks > 1
    for input_array in inputs:
        if not hasattr(input_array, "shape"):
            msg = "Op inputs must expose a shape"
            raise TypeError(msg)
        shape = tuple(int(axis_size) for axis_size in input_array.shape)
        if batched:
            if shape[0] != region.n_blocks:
                msg = (
                    f"Input batch dimension {shape[0]} does not match "
                    f"Region.n_blocks={region.n_blocks}"
                )
                raise ValueError(msg)
            block_shape = shape[1:]
        else:
            block_shape = shape
        if len(block_shape) < required.ndim:
            msg = "Op input rank is smaller than the Region rank"
            raise ValueError(msg)
        for axis, axis_slice in enumerate(required.slices):
            start, stop = _checked_bounds(axis_slice)
            if start < 0 or stop > block_shape[axis]:
                msg = "Op Region plus access pattern exceeds input bounds"
                raise ValueError(msg)


def _checked_bounds(axis_slice: slice) -> tuple[int, int]:
    if axis_slice.start is None or axis_slice.stop is None:
        msg = "Extent slices must have explicit start and stop"
        raise ValueError(msg)
    if axis_slice.step not in (None, 1):
        msg = "Extent slices do not support non-unit steps"
        raise ValueError(msg)
    return int(axis_slice.start), int(axis_slice.stop)


def _slice_length(axis_slice: slice) -> int:
    start, stop = _checked_bounds(axis_slice)
    length = stop - start
    if length < 0:
        msg = "Extent slices must be half-open with stop >= start"
        raise ValueError(msg)
    return length


__all__ = [
    "AccessPattern",
    "ComponentId",
    "Descriptor",
    "Domain",
    "Extent",
    "Map",
    "Record",
    "Region",
    "Sink",
    "Source",
    "execute_pointwise",
]
