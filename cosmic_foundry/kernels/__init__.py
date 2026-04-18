"""Kernel interface primitives."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Protocol, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.observability import get_logger

_log = get_logger(__name__)


class Backend(Enum):
    """Kernel backend identifiers."""

    JAX = "jax"


class AccessPattern(ABC):
    """Metadata describing the locality contract for an Op."""

    @abstractmethod
    def halo_width(self, axis: int) -> int:
        """Return the ghost-cell depth required on one axis."""


@dataclass(frozen=True)
class Stencil(AccessPattern):
    """Fixed-width neighborhood centered on a grid element."""

    radii: tuple[int, ...]

    @classmethod
    def seven_point(cls) -> Stencil:
        """Return the 3-D seven-point nearest-neighbor stencil."""
        return cls((1, 1, 1))

    @classmethod
    def symmetric(cls, order: int, ndim: int = 3) -> Stencil:
        """Return a symmetric stencil for an even finite-difference order."""
        if order <= 0 or order % 2 != 0:
            msg = "symmetric stencil order must be a positive even integer"
            raise ValueError(msg)
        if ndim <= 0:
            msg = "ndim must be positive"
            raise ValueError(msg)
        return cls((order // 2,) * ndim)

    def halo_width(self, axis: int) -> int:
        return self.radii[axis]


@dataclass(frozen=True)
class Extent:
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


@dataclass(frozen=True)
class Region:
    """Iteration coordinates requested by a Dispatch.

    ``n_blocks`` signals a batched region: inputs are expected to carry a
    leading batch axis of that size, and ``FlatPolicy`` lowers the kernel
    with ``jax.vmap`` so the Op remains unaware of the batch dimension.
    """

    extent: Extent
    n_blocks: int = 1

    def __post_init__(self) -> None:
        if self.n_blocks < 1:
            msg = "Region.n_blocks must be a positive integer"
            raise ValueError(msg)


class OpLike(Protocol):
    """Structural protocol for executable Ops."""

    access_pattern: AccessPattern
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    backends: frozenset[Backend]

    def __call__(self, *args: Any) -> BoundOp:
        """Bind field inputs to this Op, returning a BoundOp."""

    def _fn(self, *args: Any) -> Any:
        """Pointwise kernel: field arrays followed by index meshgrids."""


@dataclass
class BoundOp:
    """An Op with its field inputs bound, ready for dispatch.

    Created by calling an Op with its field arrays::

        bound = laplacian(phi_array)          # positional
        bound = flux(rho=rho_arr, v=v_arr)    # keyword
        Dispatch(bound, region).execute()
    """

    op: Any  # OpLike — Any avoids circular-protocol issues
    fields: dict[str, Any]  # name → array, ordered by op.reads


class Op(ABC):
    """Nominal base class for class-shaped Ops.

    Every concrete Op defines its kernel logic in ``_fn`` and declares
    ``access_pattern``, ``reads``, and ``writes`` as class attributes.
    ``__call__`` is provided here: it binds field arrays by name and
    returns a :class:`BoundOp` ready for :class:`Dispatch`.

    Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable and the
    JIT cache in :func:`_make_jit_kernel` can deduplicate compilations.
    """

    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ()
    backends: ClassVar[frozenset[Backend]] = frozenset({Backend.JAX})

    @property
    @abstractmethod
    def access_pattern(self) -> AccessPattern:
        """Return the Op locality metadata."""

    @abstractmethod
    def _fn(self, *args: Any) -> Any:
        """Pointwise kernel: field arrays followed by index meshgrids."""

    def __call__(self, *args: Any, **kwargs: Any) -> BoundOp:
        """Bind positional and keyword field arrays to this Op."""
        if len(args) > len(self.reads):
            n_reads = len(self.reads)
            name = type(self).__name__
            msg = (
                f"Op {name!r} reads {self.reads!r} ({n_reads} fields) "
                f"but received {len(args)} positional arguments"
            )
            raise TypeError(msg)
        fields: dict[str, Any] = {self.reads[i]: arg for i, arg in enumerate(args)}
        fields.update(kwargs)
        return BoundOp(op=self, fields=fields)


@dataclass(frozen=True)
class FlatPolicy:
    """Evaluate an Op at every point in a Region using JAX/XLA.

    Map:
        domain   — (k : Ω_h^ext → ℝⁿ, Ω_h^int ⊆ Ω_h^ext) — a kernel
                   and the interior region over which it is evaluated;
                   inputs must cover Ω_h^int expanded by k.access_pattern
        codomain — k(x) for x ∈ Ω_h^int — the kernel evaluated pointwise
                   over the interior; shape matches Ω_h^int
        operator — (k, Ω_h^int) ↦ jax.jit(k)(Ω_h^int);
                   when Region.n_blocks > 1 the kernel is lifted with
                   jax.vmap before JIT so the Op remains unaware of the
                   batch dimension

    Exact: Θ = ∅ — the policy introduces no approximation.
    """

    def execute(
        self,
        bound: BoundOp,
        region: Region,
    ) -> Any:
        """Execute a BoundOp over a Region with JAX/XLA.

        When ``region.n_blocks > 1`` inputs carry a leading batch axis and
        the kernel is lowered with ``jax.vmap``; the Op is unaware of that
        dimension.
        """
        op_like = bound.op
        inputs = tuple(bound.fields.values())
        _validate_op(op_like)
        _validate_region_access(region, op_like.access_pattern, inputs)

        _log.debug(
            "dispatch.execute",
            extra={
                "region_shape": list(region.extent.shape),
                "n_blocks": region.n_blocks,
            },
        )

        return _make_jit_kernel(cast(Any, op_like), region)(*inputs)


@dataclass(frozen=True)
class Dispatch:
    """One local lowering unit: a BoundOp over a Region under a Policy.

    Map:
        domain   — (k: BoundOp, Ω_h^int: Region, π: FlatPolicy) — a kernel
                   with bound field inputs, an iteration region, and an
                   execution policy
        codomain — π(k, Ω_h^int) — the kernel evaluated over the interior
        operator — execute() ↦ policy.execute(bound, region)

    Exact: Θ = ∅ — Dispatch introduces no approximation; that lives in the
    policy and Op.
    """

    # One Op per Dispatch; multi-Op fusion is deferred to the Epoch 2
    # task-graph driver, which will compose compatible Ops before lowering.
    bound: BoundOp
    region: Region
    policy: FlatPolicy = field(default_factory=FlatPolicy)

    def execute(self) -> Any:
        """Execute this Dispatch."""
        return self.policy.execute(self.bound, self.region)


@functools.lru_cache(maxsize=256)
def _make_jit_kernel(op_like: Any, region: Any) -> Callable[..., Any]:
    """Return a cached JIT-compiled kernel for *(op_like, region)*.

    Both arguments must be hashable.  ``_FuncOp`` instances satisfy this via
    ``__hash__`` / ``__eq__`` defined on the wrapper.  Class-based ``Op``
    subclasses satisfy it via object identity unless they define ``__eq__``
    without a matching ``__hash__``.  ``Region`` satisfies it as a frozen
    dataclass (with ``Extent.__hash__`` converting slice bounds to a tuple).

    Caching ensures that repeated ``FlatPolicy.execute`` calls with the same
    Op and Region reuse the same compiled XLA computation rather than
    re-tracing on every invocation.
    """
    if region.n_blocks > 1:

        @jax.jit
        def apply_batched(*jit_inputs: Any) -> Any:
            indices = _region_indices(region)

            def single_block(*block_inputs: Any) -> Any:
                return op_like._fn(*block_inputs, *indices)

            return jax.vmap(single_block)(*jit_inputs)

        return cast(Callable[..., Any], apply_batched)

    @jax.jit
    def apply(*jit_inputs: Any) -> Any:
        indices = _region_indices(region)
        return op_like._fn(*jit_inputs, *indices)

    return cast(Callable[..., Any], apply)


def _region_indices(region: Region) -> tuple[jax.Array, ...]:
    axes = []
    for axis_slice in region.extent.slices:
        start, stop = _checked_bounds(axis_slice)
        axes.append(jnp.arange(start, stop))
    return tuple(jnp.meshgrid(*axes, indexing="ij"))


def _validate_op(op_like: Any) -> None:
    if not hasattr(op_like, "access_pattern"):
        msg = "Op must declare access_pattern metadata"
        raise TypeError(msg)
    if Backend.JAX not in op_like.backends:
        msg = "FlatPolicy requires an Op that supports the JAX backend"
        raise ValueError(msg)


def _validate_region_access(
    region: Region,
    access_pattern: AccessPattern,
    inputs: tuple[Any, ...],
) -> None:
    required = region.extent.expand(access_pattern)
    batched = region.n_blocks > 1
    for input_array in inputs:
        if not hasattr(input_array, "shape"):
            msg = "Dispatch inputs must expose a shape"
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
            msg = "Dispatch input rank is smaller than the Region rank"
            raise ValueError(msg)
        for axis, axis_slice in enumerate(required.slices):
            start, stop = _checked_bounds(axis_slice)
            if start < 0 or stop > block_shape[axis]:
                msg = "Dispatch Region plus access pattern exceeds input bounds"
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
    "Backend",
    "BoundOp",
    "Dispatch",
    "Extent",
    "FlatPolicy",
    "Op",
    "OpLike",
    "Region",
    "Stencil",
]
