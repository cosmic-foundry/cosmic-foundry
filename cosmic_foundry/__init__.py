"""Cosmic Foundry — high-performance computational astrophysics engine."""

import jax

# ADR-0009: the package guarantees float64 precision at import time.
# Must be set before any JAX computation is JIT-compiled.
jax.config.update("jax_enable_x64", True)

from cosmic_foundry._version import __version__  # noqa: E402
from cosmic_foundry.computation.kernels import execute_pointwise  # noqa: E402
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
from cosmic_foundry.discretization import Discretization  # noqa: E402
from cosmic_foundry.field import (  # noqa: E402
    ContinuousField,
    Field,
    ScalarField,
    TensorField,
)
from cosmic_foundry.function import Function  # noqa: E402
from cosmic_foundry.indexed_set import IndexedSet  # noqa: E402
from cosmic_foundry.io import (  # noqa: E402
    HAS_PARALLEL_HDF5,
    MergeRankFiles,
    WriteArray,
    merge_rank_files,
    write_array,
)
from cosmic_foundry.located_discretization import LocatedDiscretization  # noqa: E402
from cosmic_foundry.mesh import (  # noqa: E402
    PartitionDomain,
    Patch,
    covers,
    fill_halo,
    partition_domain,
)
from cosmic_foundry.modal_discretization import ModalDiscretization  # noqa: E402
from cosmic_foundry.observability import configure, get_logger  # noqa: E402
from cosmic_foundry.pseudo_riemannian_manifold import (  # noqa: E402
    PseudoRiemannianManifold,
)
from cosmic_foundry.record import Array, ComponentId, Placement, Record  # noqa: E402
from cosmic_foundry.riemannian_manifold import RiemannianManifold  # noqa: E402
from cosmic_foundry.set import Set  # noqa: E402
from cosmic_foundry.sink import Sink  # noqa: E402
from cosmic_foundry.smooth_manifold import SmoothManifold  # noqa: E402
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
    "Field",
    "ScalarField",
    "TensorField",
    # io
    "HAS_PARALLEL_HDF5",
    "MergeRankFiles",
    "WriteArray",
    "merge_rank_files",
    "write_array",
    # kernels
    "AccessPattern",
    "Array",
    "ComponentId",
    "Descriptor",
    "Placement",
    "Discretization",
    "IndexedSet",
    "LocatedDiscretization",
    "ModalDiscretization",
    "PseudoRiemannianManifold",
    "RiemannianManifold",
    "Set",
    "SmoothManifold",
    "Extent",
    "Function",
    "Record",
    "Region",
    "Sink",
    "Source",
    "execute_pointwise",
    # mesh
    "Patch",
    "PartitionDomain",
    "covers",
    "fill_halo",
    "partition_domain",
    # observability
    "configure",
    "get_logger",
]
