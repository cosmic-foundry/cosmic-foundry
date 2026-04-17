"""Tests for the uniform structured mesh."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_foundry.kernels import Extent
from cosmic_foundry.mesh import Block, BlockId, UniformGrid


class TestBlock:
    def _make_block(self) -> Block:
        return Block(
            block_id=BlockId(0),
            index_extent=Extent((slice(0, 8), slice(0, 8), slice(0, 8))),
            origin=(0.0625, 0.0625, 0.0625),
            cell_spacing=(0.125, 0.125, 0.125),
        )

    def test_shape_and_ndim(self):
        b = self._make_block()
        assert b.shape == (8, 8, 8)
        assert b.ndim == 3

    def test_cell_centers_values(self):
        h = 0.25
        block = Block(
            block_id=BlockId(0),
            index_extent=Extent((slice(0, 4),)),
            origin=(0.5 * h,),
            cell_spacing=(h,),
        )
        centers = np.asarray(block.cell_centers(0))
        expected = np.array([0.125, 0.375, 0.625, 0.875])
        np.testing.assert_allclose(centers, expected)

    def test_cell_centers_spacing(self):
        b = self._make_block()
        centers = np.asarray(b.cell_centers(0))
        diffs = np.diff(centers)
        np.testing.assert_allclose(diffs, b.cell_spacing[0])


class TestUniformGrid:
    def _make_2d(self, **kwargs) -> UniformGrid:
        defaults = dict(
            domain_origin=(0.0, 0.0),
            domain_size=(1.0, 1.0),
            n_cells=(8, 8),
            blocks_per_axis=(2, 2),
            n_ranks=2,
        )
        defaults.update(kwargs)
        return UniformGrid.create(**defaults)

    def test_block_count(self):
        assert len(self._make_2d().blocks) == 4  # 2×2

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            UniformGrid.create(
                domain_origin=(0.0,),
                domain_size=(1.0,),
                n_cells=(7,),
                blocks_per_axis=(2,),
                n_ranks=1,
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            UniformGrid.create(
                domain_origin=(0.0,),
                domain_size=(1.0, 1.0),
                n_cells=(8,),
                blocks_per_axis=(2,),
                n_ranks=1,
            )

    def test_index_extents_tile_domain(self):
        n_cells = (8, 12)
        grid = UniformGrid.create(
            domain_origin=(0.0, 0.0),
            domain_size=(1.0, 1.0),
            n_cells=n_cells,
            blocks_per_axis=(2, 3),
            n_ranks=1,
        )
        covered = np.zeros(n_cells, dtype=int)
        for block in grid.blocks:
            covered[block.index_extent.slices] += 1
        assert np.all(covered == 1), "blocks must tile the domain exactly once"

    def test_cell_centers_cover_domain(self):
        """First and last cell centers sit h/2 from the domain edges."""
        grid = UniformGrid.create(
            domain_origin=(0.0,),
            domain_size=(1.0,),
            n_cells=(4,),
            blocks_per_axis=(2,),
            n_ranks=1,
        )
        all_centers = np.concatenate(
            [np.asarray(b.cell_centers(0)) for b in grid.blocks]
        )
        h = 0.25
        assert all_centers[0] == pytest.approx(0.5 * h)
        assert all_centers[-1] == pytest.approx(1.0 - 0.5 * h)

    def test_round_robin_placement(self):
        grid = UniformGrid.create(
            domain_origin=(0.0,),
            domain_size=(1.0,),
            n_cells=(8,),
            blocks_per_axis=(4,),
            n_ranks=2,
        )
        assert grid.owner(BlockId(0)) == 0
        assert grid.owner(BlockId(1)) == 1
        assert grid.owner(BlockId(2)) == 0
        assert grid.owner(BlockId(3)) == 1

    def test_blocks_for_rank_partition(self):
        grid = self._make_2d(n_ranks=2)
        r0 = grid.blocks_for_rank(0)
        r1 = grid.blocks_for_rank(1)
        assert len(r0) + len(r1) == len(grid.blocks)
        assert set(b.block_id for b in r0).isdisjoint(b.block_id for b in r1)

    def test_block_lookup(self):
        grid = self._make_2d()
        for block in grid.blocks:
            assert grid.block(block.block_id) is block

    def test_3d_tiling(self):
        n_cells = (8, 8, 8)
        grid = UniformGrid.create(
            domain_origin=(0.0, 0.0, 0.0),
            domain_size=(1.0, 1.0, 1.0),
            n_cells=n_cells,
            blocks_per_axis=(2, 2, 2),
            n_ranks=4,
        )
        assert len(grid.blocks) == 8
        covered = np.zeros(n_cells, dtype=int)
        for block in grid.blocks:
            covered[block.index_extent.slices] += 1
        assert np.all(covered == 1)

    def test_cell_spacing_uniform(self):
        grid = self._make_2d()
        for block in grid.blocks:
            assert block.cell_spacing == (0.125, 0.125)
