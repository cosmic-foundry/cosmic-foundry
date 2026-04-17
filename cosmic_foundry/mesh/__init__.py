"""Uniform structured mesh."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, NewType

from cosmic_foundry.kernels import Extent

BlockId = NewType("BlockId", int)


@dataclass(frozen=True)
class Block:
    """One contiguous patch of uniformly-spaced cells.

    Owns topology and coordinate metadata only; array payloads
    live in Field/FieldSegment.
    """

    block_id: BlockId
    index_extent: Extent
    origin: tuple[float, ...]  # physical coord of first cell center
    cell_spacing: tuple[float, ...]  # h_i along each axis

    @property
    def ndim(self) -> int:
        return self.index_extent.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.index_extent.shape

    def cell_centers(self, axis: int) -> Any:
        """1-D coordinate array of cell centers along *axis*."""
        import jax.numpy as jnp

        return (
            self.origin[axis] + jnp.arange(self.shape[axis]) * self.cell_spacing[axis]
        )


@dataclass(frozen=True)
class UniformGrid:
    """Uniform domain partitioned into a structured layout of blocks.

    Blocks are enumerated in C order (last axis varies fastest) and
    distributed across ranks round-robin by flat index.
    """

    blocks: tuple[Block, ...]
    rank_map: tuple[int, ...]  # rank_map[block_id] → owning rank

    def block(self, block_id: BlockId) -> Block:
        return self.blocks[block_id]

    def owner(self, block_id: BlockId) -> int:
        return self.rank_map[block_id]

    def blocks_for_rank(self, rank: int) -> tuple[Block, ...]:
        return tuple(b for b in self.blocks if self.rank_map[b.block_id] == rank)

    @classmethod
    def create(
        cls,
        *,
        domain_origin: tuple[float, ...],
        domain_size: tuple[float, ...],
        n_cells: tuple[int, ...],
        blocks_per_axis: tuple[int, ...],
        n_ranks: int,
    ) -> UniformGrid:
        """Partition a uniform domain into blocks and assign them to ranks.

        Raises ValueError if any axis of n_cells is not divisible by the
        corresponding blocks_per_axis entry.
        """
        ndim = len(n_cells)
        if not (len(domain_origin) == len(domain_size) == len(blocks_per_axis) == ndim):
            raise ValueError(
                "domain_origin, domain_size, n_cells, and blocks_per_axis "
                "must all have the same length"
            )
        for i in range(ndim):
            if n_cells[i] % blocks_per_axis[i] != 0:
                raise ValueError(
                    f"n_cells[{i}]={n_cells[i]} is not divisible by "
                    f"blocks_per_axis[{i}]={blocks_per_axis[i]}"
                )

        h = tuple(domain_size[i] / n_cells[i] for i in range(ndim))
        cpb = tuple(n_cells[i] // blocks_per_axis[i] for i in range(ndim))

        blocks: list[Block] = []
        rank_map: list[int] = []

        for flat_id, multi_idx in enumerate(
            itertools.product(*(range(blocks_per_axis[i]) for i in range(ndim)))
        ):
            starts = tuple(multi_idx[i] * cpb[i] for i in range(ndim))
            stops = tuple(starts[i] + cpb[i] for i in range(ndim))
            index_extent = Extent(
                tuple(slice(starts[i], stops[i]) for i in range(ndim))
            )
            origin = tuple(
                domain_origin[i] + (starts[i] + 0.5) * h[i] for i in range(ndim)
            )
            blocks.append(
                Block(
                    block_id=BlockId(flat_id),
                    index_extent=index_extent,
                    origin=origin,
                    cell_spacing=h,
                )
            )
            rank_map.append(flat_id % n_ranks)

        return cls(blocks=tuple(blocks), rank_map=tuple(rank_map))
