"""Cosmic Foundry — high-performance computational astrophysics engine."""

import jax

# ADR-0009: the package guarantees float64 precision at import time.
# Must be set before any JAX computation is JIT-compiled.
jax.config.update("jax_enable_x64", True)

from cosmic_foundry._version import __version__  # noqa: E402
from cosmic_foundry.fields import (  # noqa: E402
    Field,
    FieldSegment,
    Placement,
    SegmentId,
    allocate_field,
)
from cosmic_foundry.halo import HaloFillFence, HaloFillPolicy  # noqa: E402
from cosmic_foundry.io import (  # noqa: E402
    HAS_PARALLEL_HDF5,
    merge_rank_files,
    write_array,
)
from cosmic_foundry.kernels import (  # noqa: E402
    AccessPattern,
    Backend,
    BoundOp,
    Dispatch,
    Extent,
    FlatPolicy,
    Op,
    OpLike,
    Region,
    Stencil,
    op,
)
from cosmic_foundry.mesh import Block, BlockId, UniformGrid  # noqa: E402
from cosmic_foundry.observability import configure, get_logger  # noqa: E402

__all__ = [
    "__version__",
    # fields
    "Field",
    "FieldSegment",
    "Placement",
    "SegmentId",
    "allocate_field",
    # halo
    "HaloFillFence",
    "HaloFillPolicy",
    # io
    "HAS_PARALLEL_HDF5",
    "merge_rank_files",
    "write_array",
    # kernels
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
    "op",
    # mesh
    "Block",
    "BlockId",
    "UniformGrid",
    # observability
    "configure",
    "get_logger",
]
