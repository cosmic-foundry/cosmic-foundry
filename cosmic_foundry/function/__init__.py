"""Function ABC and pointwise execution engine."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.descriptor import AccessPattern, Region, _checked_bounds


class Function(ABC):
    """Abstract base for all function classes: f: A × Θ → B.

    Every concrete Function subclass carries a ``Function:`` block in its
    class docstring specifying domain, codomain, operator, Θ, and
    approximation order p.  Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function and return the result."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to execute(); lets a Function instance be used as a callable."""
        return self.execute(*args, **kwargs)


def execute_pointwise(
    fn: Any,
    region: Region,
    *field_arrays: Any,
) -> Any:
    """Apply fn._fn over region with JAX JIT and input validation.

    ``fn`` must be hashable (for JIT caching) and expose:

    - ``_fn(*field_arrays, *index_meshgrids) -> scalar``
    - ``access_pattern: AccessPattern``

    When ``region.n_blocks > 1`` the kernel is lifted with ``jax.vmap``
    so ``_fn`` remains unaware of the batch dimension.
    """
    _validate_region_access(region, fn.access_pattern, field_arrays)
    return _make_jit_kernel(cast(Any, fn), region)(*field_arrays)


@functools.lru_cache(maxsize=256)
def _make_jit_kernel(fn: Any, region: Any) -> Callable[..., Any]:
    """Return a cached JIT-compiled kernel for *(fn, region)*.

    Both arguments must be hashable.  Frozen dataclasses satisfy this via
    their auto-generated ``__hash__``.  ``Region`` satisfies it as a frozen
    dataclass (with ``Extent.__hash__`` converting slice bounds to a tuple).

    Caching ensures repeated ``execute_pointwise`` calls with the same
    fn and Region reuse the same compiled XLA computation.
    """
    if region.n_blocks > 1:

        @jax.jit
        def apply_batched(*jit_inputs: Any) -> Any:
            indices = _region_indices(region)

            def single_block(*block_inputs: Any) -> Any:
                return fn._fn(*block_inputs, *indices)

            return jax.vmap(single_block)(*jit_inputs)

        return cast(Callable[..., Any], apply_batched)

    @jax.jit
    def apply(*jit_inputs: Any) -> Any:
        indices = _region_indices(region)
        return fn._fn(*jit_inputs, *indices)

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
            msg = "Function inputs must expose a shape"
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
            msg = "Function input rank is smaller than the Region rank"
            raise ValueError(msg)
        for axis, axis_slice in enumerate(required.slices):
            start, stop = _checked_bounds(axis_slice)
            if start < 0 or stop > block_shape[axis]:
                msg = "Function Region plus access pattern exceeds input bounds"
                raise ValueError(msg)


__all__ = [
    "Function",
    "execute_pointwise",
]
