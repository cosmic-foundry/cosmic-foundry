"""Cosmic Foundry — high-performance computational astrophysics engine."""

import jax

# ADR-0009: the package guarantees float64 precision at import time.
# Must be set before any JAX computation is JIT-compiled.
jax.config.update("jax_enable_x64", True)

from cosmic_foundry._version import __version__  # noqa: E402
from cosmic_foundry.descriptor import (  # noqa: E402
    AccessPattern,
    Descriptor,
    Extent,
    Region,
)
from cosmic_foundry.diagnostics import (  # noqa: E402
    CollectDiagnostics,
    DiagnosticRecord,
    DiagnosticReducer,
    GlobalSum,
    NullDiagnosticSink,
    TabSeparatedDiagnosticSink,
    collect_diagnostics,
    global_sum,
)
from cosmic_foundry.domain import Domain  # noqa: E402
from cosmic_foundry.field import (  # noqa: E402
    ContinuousField,
    DiscreteField,
    Field,
    Placement,
)
from cosmic_foundry.io import (  # noqa: E402
    HAS_PARALLEL_HDF5,
    MergeRankFiles,
    WriteArray,
    merge_rank_files,
    write_array,
)
from cosmic_foundry.map import Map, execute_pointwise  # noqa: E402
from cosmic_foundry.mesh import (  # noqa: E402
    Block,
    DistributedField,
    FieldDiscretization,
    FieldSegment,
    PartitionDomain,
    UniformGrid,
    field_discretization,
    partition_domain,
)
from cosmic_foundry.observability import configure, get_logger  # noqa: E402
from cosmic_foundry.record import ComponentId, Record  # noqa: E402
from cosmic_foundry.sink import Sink  # noqa: E402
from cosmic_foundry.source import Source  # noqa: E402

__all__ = [
    "__version__",
    # diagnostics
    "CollectDiagnostics",
    "DiagnosticRecord",
    "DiagnosticReducer",
    "GlobalSum",
    "NullDiagnosticSink",
    "TabSeparatedDiagnosticSink",
    "collect_diagnostics",
    "global_sum",
    # fields
    "ContinuousField",
    "DiscreteField",
    "Field",
    "FieldDiscretization",
    "Placement",
    # io
    "HAS_PARALLEL_HDF5",
    "MergeRankFiles",
    "WriteArray",
    "merge_rank_files",
    "write_array",
    # kernels
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
    # mesh
    "Block",
    "DistributedField",
    "FieldDiscretization",
    "FieldSegment",
    "PartitionDomain",
    "UniformGrid",
    "field_discretization",
    "partition_domain",
    # observability
    "configure",
    "get_logger",
]
