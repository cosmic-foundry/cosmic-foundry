"""Kernel interface primitives."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import jax
import jax.numpy as jnp

from cosmic_foundry.observability import get_logger

_log = get_logger(__name__)


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


class AccessPattern(Descriptor):
    """Metadata describing the locality contract for an Op."""

    @abstractmethod
    def halo_width(self, axis: int) -> int:
        """Return the ghost-cell depth required on one axis."""

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this access pattern."""
        return {"type": type(self).__name__}


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

    def as_dict(self) -> dict[str, Any]:
        return {"type": "Stencil", "radii": list(self.radii)}


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

    def as_dict(self) -> dict[str, Any]:
        return {"extent": self.extent.as_dict(), "n_blocks": self.n_blocks}


class Record(ABC):
    """Abstract base for all record types: lightweight immutable value objects
    that are *about* the simulation rather than *being* simulation state.

    Records are internal objects produced or consumed at the semantic layer —
    summaries, identifiers, and provenance metadata. They are distinct from
    Fields (which ARE simulation state) and from the external representations
    (bytes, files) that Sources and Sinks translate to and from.

    Every Record must be serializable to a plain dict.
    """

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this record."""


@dataclass(frozen=True)
class ComponentId(Record):
    """Opaque integer identifier for a named simulation component.

    Used wherever a typed, hashable, serializable integer key is needed —
    mesh blocks, field segments, and any future entity type.  A single class
    avoids redundant id types for concepts that are structurally identical.
    """

    value: int

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value}


class Domain(ABC):
    """Abstract base for all domain types: the set D over which fields are defined.

    A domain is the input space of a field f: D → ℝ. Domains differ in their
    representation (continuous vs. discrete) and their nature (physical space,
    thermodynamic state space, etc.). Every Field has a Domain; a Domain is
    not itself a Field.

    The one universal property of a domain is its dimensionality.
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of dimensions of this domain."""


class Map(ABC):
    """Abstract base for all map classes: M: A × Θ → B.

    Every concrete Map subclass carries a ``Map:`` block in its class
    docstring specifying domain, codomain, operator, Θ, and approximation
    order p.  Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the map and return the result."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to execute(); lets a Map instance be used as a callable."""
        return self.execute(*args, **kwargs)


class Source(ABC):
    """Abstract base for all source classes: R: external state → B.

    Every concrete Source subclass carries a ``Source:`` block in its class
    docstring specifying the external state consumed (origin) and the value
    produced.  Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Read from external state and return the result."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to execute()."""
        return self.execute(*args, **kwargs)


class Sink(ABC):
    """Abstract base for all sink classes: S: A → external state.

    Every concrete Sink subclass carries a ``Sink:`` block in its class
    docstring specifying the domain consumed and the external effect produced.
    Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable.
    """

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Consume input and materialise it into external state."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to execute()."""
        return self.execute(*args, **kwargs)


class Op(Map, Descriptor):
    """Abstract base for class-shaped kernels.

    Every concrete Op defines its pointwise kernel logic in ``_fn`` and
    declares ``access_pattern``, ``reads``, and ``writes`` as class
    attributes. ``execute`` (and thus ``__call__``) runs the kernel
    directly over the supplied field arrays and region via a Policy.

    Subclasses that carry no parameters should use
    ``@dataclass(frozen=True)`` so that instances are hashable and the
    JIT cache in :func:`_make_jit_kernel` can deduplicate compilations.

    Map:
        domain   — (*field_arrays, region: Region) — field arrays in
                   reads order, covering region expanded by access_pattern
        codomain — pointwise kernel result over region.extent
        operator — execute(*field_arrays, region) ↦ policy(self, region)
    """

    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ()

    @property
    @abstractmethod
    def access_pattern(self) -> AccessPattern:
        """Return the Op locality metadata."""

    @abstractmethod
    def _fn(self, *args: Any) -> Any:
        """Pointwise kernel: field arrays followed by index meshgrids."""

    def execute(self, *field_arrays: Any, region: Region, policy: Any = None) -> Any:
        """Run this Op over *region* via *policy* (default: FlatPolicy)."""
        p: FlatPolicy = policy if policy is not None else FlatPolicy()
        return p.execute(self, *field_arrays, region=region)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "access_pattern": self.access_pattern.as_dict(),
        }


@dataclass(frozen=True)
class FlatPolicy(Map):
    """Evaluate an Op at every point in a Region using JAX/XLA.

    Map:
        domain   — (k: Op, *field_arrays, region: Region) — a kernel,
                   its field inputs in reads order, and the interior
                   region; inputs must cover region expanded by
                   k.access_pattern
        codomain — k(x) for x ∈ region.extent — the kernel evaluated
                   pointwise over the interior
        operator — (k, fields, Ω_h^int) ↦ jax.jit(k)(Ω_h^int);
                   when region.n_blocks > 1 the kernel is lifted with
                   jax.vmap before JIT so the Op remains unaware of the
                   batch dimension

    Exact: Θ = ∅ — the policy introduces no approximation.
    """

    def execute(self, op: Any, *field_arrays: Any, region: Region) -> Any:
        """Execute *op* over *region* with JAX/XLA."""
        _validate_op(op)
        _validate_region_access(region, op.access_pattern, field_arrays)

        _log.debug(
            "dispatch.execute",
            extra={
                "region_shape": list(region.extent.shape),
                "n_blocks": region.n_blocks,
            },
        )

        return _make_jit_kernel(cast(Any, op), region)(*field_arrays)


@dataclass
class Dispatch(Map, Descriptor):
    """One local execution unit: an Op over a Region under a Policy.

    Carries the full execution plan as an inspectable record. A future
    task-graph driver can collect a sequence of Dispatches, analyze
    reads/writes for fusion compatibility, and lower fused kernels before
    calling execute(). One Op per Dispatch; multi-Op fusion is the
    driver's responsibility.

    Map:
        domain   — (k: Op, {f_i}, Ω_h^int: Region, π: FlatPolicy) — a
                   kernel, its field inputs, an iteration region, and a
                   policy
        codomain — π(k, {f_i}, Ω_h^int) — the kernel evaluated over
                   the interior
        operator — execute() ↦ op.execute(*fields.values(), region,
                   policy=policy)

    Exact: Θ = ∅ — Dispatch introduces no approximation.
    """

    op: Any  # Op instance
    fields: dict[str, Any]  # name → array, ordered by op.reads
    region: Region
    policy: FlatPolicy = field(default_factory=FlatPolicy)

    def execute(self) -> Any:
        """Execute this Dispatch."""
        return self.op.execute(
            *self.fields.values(), region=self.region, policy=self.policy
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op.as_dict(),
            "fields": list(self.fields.keys()),
            "region": self.region.as_dict(),
            "policy": type(self.policy).__name__,
        }


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
    "ComponentId",
    "Descriptor",
    "Dispatch",
    "Domain",
    "Extent",
    "FlatPolicy",
    "Map",
    "Op",
    "Record",
    "Region",
    "Sink",
    "Source",
    "Stencil",
]
