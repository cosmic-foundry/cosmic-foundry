"""Tests for allocate_field: Field allocation from UniformGrid blocks."""

from __future__ import annotations

import jax.numpy as jnp

from cosmic_foundry.fields import Field, SegmentId, allocate_field
from cosmic_foundry.kernels import Extent, Stencil
from cosmic_foundry.mesh import UniformGrid


def _grid_1d(n_cells: int, n_blocks: int, n_ranks: int = 1) -> UniformGrid:
    return UniformGrid.create(
        domain_origin=(0.0,),
        domain_size=(1.0,),
        n_cells=(n_cells,),
        blocks_per_axis=(n_blocks,),
        n_ranks=n_ranks,
    )


def _grid_2d(
    n_cells: tuple[int, int], blocks_per_axis: tuple[int, int], n_ranks: int = 1
) -> UniformGrid:
    return UniformGrid.create(
        domain_origin=(0.0, 0.0),
        domain_size=(1.0, 1.0),
        n_cells=n_cells,
        blocks_per_axis=blocks_per_axis,
        n_ranks=n_ranks,
    )


class TestAllocateFieldShape:
    def test_zero_halo_payload_shape_matches_block(self) -> None:
        grid = _grid_1d(8, 2)
        field = allocate_field("phi", grid, Stencil((0,)))
        for block in grid.blocks:
            seg = field.segment(SegmentId(int(block.block_id)))
            assert seg.payload.shape == block.shape

    def test_halo_width_1_expands_payload(self) -> None:
        grid = _grid_1d(8, 2)
        access = Stencil((1,))
        field = allocate_field("phi", grid, access)
        for block in grid.blocks:
            seg = field.segment(SegmentId(int(block.block_id)))
            expected_shape = tuple(n + 2 for n in block.shape)
            assert seg.payload.shape == expected_shape

    def test_2d_halo_expands_all_axes(self) -> None:
        grid = _grid_2d((8, 12), (2, 3))
        access = Stencil((2, 1))
        field = allocate_field("phi", grid, access)
        for block in grid.blocks:
            seg = field.segment(SegmentId(int(block.block_id)))
            expected = (block.shape[0] + 4, block.shape[1] + 2)
            assert seg.payload.shape == expected


class TestAllocateFieldExtent:
    def test_segment_extent_is_halo_expanded_block_extent(self) -> None:
        grid = _grid_1d(8, 2)
        access = Stencil((1,))
        field = allocate_field("phi", grid, access)
        for block in grid.blocks:
            seg = field.segment(SegmentId(int(block.block_id)))
            expected = block.index_extent.expand(access)
            assert seg.extent == expected

    def test_segment_interior_extent_matches_block_index_extent(self) -> None:
        grid = _grid_1d(8, 2)
        field = allocate_field("phi", grid, Stencil((1,)))
        for block in grid.blocks:
            seg = field.segment(SegmentId(int(block.block_id)))
            assert seg.interior_extent == block.index_extent

    def test_zero_halo_extent_matches_block_index_extent(self) -> None:
        grid = _grid_1d(8, 2)
        field = allocate_field("phi", grid, Stencil((0,)))
        for block in grid.blocks:
            seg = field.segment(SegmentId(int(block.block_id)))
            assert seg.extent == block.index_extent


class TestAllocateFieldPayload:
    def test_payload_is_zero(self) -> None:
        grid = _grid_1d(8, 2)
        field = allocate_field("phi", grid, Stencil((1,)))
        for seg in field.segments:
            assert jnp.all(seg.payload == 0.0)

    def test_payload_dtype_is_float64(self) -> None:
        grid = _grid_1d(8, 2)
        field = allocate_field("phi", grid, Stencil((1,)))
        for seg in field.segments:
            assert seg.payload.dtype == jnp.float64


class TestAllocateFieldIdentity:
    def test_segment_id_matches_block_id(self) -> None:
        grid = _grid_1d(8, 4)
        field = allocate_field("phi", grid, Stencil((1,)))
        seg_ids = {seg.segment_id for seg in field.segments}
        block_ids = {SegmentId(int(b.block_id)) for b in grid.blocks}
        assert seg_ids == block_ids

    def test_segment_count_matches_block_count(self) -> None:
        grid = _grid_2d((8, 8), (2, 2))
        field = allocate_field("phi", grid, Stencil((1, 1)))
        assert len(field.segments) == len(grid.blocks)

    def test_field_name_preserved(self) -> None:
        grid = _grid_1d(8, 1)
        field = allocate_field("density", grid, Stencil((0,)))
        assert field.name == "density"


class TestAllocateFieldPlacement:
    def test_single_rank_all_segments_owned_by_rank_0(self) -> None:
        grid = _grid_1d(8, 4, n_ranks=1)
        field = allocate_field("phi", grid, Stencil((1,)))
        for seg in field.segments:
            assert field.placement.owner(seg.segment_id) == 0

    def test_multi_rank_placement_mirrors_grid_rank_map(self) -> None:
        grid = _grid_1d(8, 4, n_ranks=2)
        field = allocate_field("phi", grid, Stencil((1,)))
        for block in grid.blocks:
            seg_id = SegmentId(int(block.block_id))
            assert field.placement.owner(seg_id) == grid.owner(block.block_id)

    def test_2d_multi_rank_placement(self) -> None:
        grid = _grid_2d((8, 8), (2, 2), n_ranks=3)
        field = allocate_field("phi", grid, Stencil((1, 1)))
        for block in grid.blocks:
            seg_id = SegmentId(int(block.block_id))
            assert field.placement.owner(seg_id) == grid.owner(block.block_id)


class TestAllocateFieldCovers:
    def test_field_covers_each_block_interior(self) -> None:
        grid = _grid_2d((8, 8), (2, 2))
        field = allocate_field("phi", grid, Stencil((1, 1)))
        for block in grid.blocks:
            assert field.covers(block.index_extent)

    def test_field_covers_halo_expanded_extent(self) -> None:
        access = Stencil((1,))
        grid = _grid_1d(8, 2)
        field = allocate_field("phi", grid, access)
        for block in grid.blocks:
            halo_ext = block.index_extent.expand(access)
            assert field.covers(halo_ext)

    def test_field_does_not_cover_full_domain_when_multiblock(self) -> None:
        # Each segment only covers its own block (+ halo), not the whole domain.
        # The union covers the full domain but a single segment does not.
        grid = _grid_1d(8, 2, n_ranks=1)
        field = allocate_field("phi", grid, Stencil((1,)))
        full = Extent((slice(0, 8),))
        # Each individual segment covers only 4 interior cells (+halo), not all 8.
        for seg in field.segments:
            single_seg_field = Field(field.name, (seg,), field.placement)
            assert not single_seg_field.covers(full)
