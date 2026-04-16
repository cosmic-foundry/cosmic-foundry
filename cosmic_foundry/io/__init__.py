"""HDF5 I/O helpers for Cosmic Foundry (Epoch 1).

Two functions cover the Epoch 1 write path:

- :func:`write_array` — write a single array (JAX or NumPy) to a new HDF5
  file.  When parallel HDF5 is available this can be called collectively; on
  serial h5py builds each rank writes its own file and
  :func:`merge_rank_files` assembles the result.

- :func:`merge_rank_files` — concatenate a sequence of per-rank HDF5 files
  along one axis into a single output file.  This is the post-processing
  merge step used when parallel HDF5 is not available.

Parallel HDF5 availability is detected at runtime via
``h5py.h5.get_config().mpi``; the result is exposed as :data:`HAS_PARALLEL_HDF5`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from cosmic_foundry.observability import get_logger

_log = get_logger(__name__)

#: True when h5py was built with MPI support (parallel HDF5 available).
HAS_PARALLEL_HDF5: bool = bool(h5py.h5.get_config().mpi)


def write_array(
    path: str | Path,
    array: Any,
    dataset: str = "data",
) -> None:
    """Write *array* to a new HDF5 file at *path* under dataset name *dataset*.

    *array* may be a JAX array or a NumPy array; it is converted to NumPy
    before the write so no JAX-specific serialization is required.

    Parameters
    ----------
    path:
        Destination file path.  Any existing file at this path is overwritten.
    array:
        Array to write.  Converted to ``numpy.ndarray`` if necessary.
    dataset:
        HDF5 dataset name inside the file.
    """
    np_array = np.asarray(array)
    dest = Path(path)
    _log.debug(
        "io.write_array",
        extra={"path": str(dest), "dataset": dataset, "shape": list(np_array.shape)},
    )
    with h5py.File(dest, "w") as f:
        f.create_dataset(dataset, data=np_array)


def merge_rank_files(
    rank_paths: Sequence[str | Path],
    output_path: str | Path,
    dataset: str = "data",
    *,
    axis: int = 0,
) -> None:
    """Concatenate per-rank HDF5 files into a single output file.

    Each file in *rank_paths* must contain a dataset named *dataset*.  The
    arrays are concatenated in order along *axis* and written to *output_path*.

    This is the post-processing merge step for the ``HAS_PARALLEL_HDF5=False``
    write path.

    Parameters
    ----------
    rank_paths:
        Ordered sequence of per-rank file paths (rank 0 first).
    output_path:
        Destination file path for the merged result.
    dataset:
        HDF5 dataset name to read from each rank file and write to the output.
    axis:
        Axis along which to concatenate.
    """
    if not rank_paths:
        msg = "rank_paths must contain at least one file"
        raise ValueError(msg)

    slabs = []
    for p in rank_paths:
        with h5py.File(Path(p), "r") as f:
            slabs.append(f[dataset][()])

    merged = np.concatenate(slabs, axis=axis)
    dest = Path(output_path)
    _log.debug(
        "io.merge_rank_files",
        extra={
            "n_ranks": len(rank_paths),
            "output_path": str(dest),
            "dataset": dataset,
            "merged_shape": list(merged.shape),
        },
    )
    with h5py.File(dest, "w") as f:
        f.create_dataset(dataset, data=merged)


__all__ = [
    "HAS_PARALLEL_HDF5",
    "merge_rank_files",
    "write_array",
]
