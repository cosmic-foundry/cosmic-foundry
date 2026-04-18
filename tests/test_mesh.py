"""Tests for the uniform structured mesh: Block, partition_domain, covers."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_foundry.descriptor import Extent
from cosmic_foundry.mesh import Block, covers, partition_domain
from cosmic_foundry.record import Array, ComponentId


class TestBlock:
    def _make_block(self) -> Block:
        return Block(
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


class TestPartitionDomain:
    def _make_2d(self, **kwargs) -> Array[Block]:
        defaults = dict(
            domain_origin=(0.0, 0.0),
            domain_size=(1.0, 1.0),
            n_cells=(8, 8),
            blocks_per_axis=(2, 2),
            n_ranks=2,
        )
        defaults.update(kwargs)
        return partition_domain(**defaults)

    def test_block_count(self):
        mesh = self._make_2d()
        assert len(mesh.elements) == 4  # 2×2

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            partition_domain(
                domain_origin=(0.0,),
                domain_size=(1.0,),
                n_cells=(7,),
                blocks_per_axis=(2,),
                n_ranks=1,
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            partition_domain(
                domain_origin=(0.0,),
                domain_size=(1.0, 1.0),
                n_cells=(8,),
                blocks_per_axis=(2,),
                n_ranks=1,
            )

    def test_index_extents_tile_domain(self):
        n_cells = (8, 12)
        mesh = partition_domain(
            domain_origin=(0.0, 0.0),
            domain_size=(1.0, 1.0),
            n_cells=n_cells,
            blocks_per_axis=(2, 3),
            n_ranks=1,
        )
        covered = np.zeros(n_cells, dtype=int)
        for block in mesh.elements:
            covered[block.index_extent.slices] += 1
        assert np.all(covered == 1), "blocks must tile the domain exactly once"

    def test_cell_centers_cover_domain(self):
        """First and last cell centers sit h/2 from the domain edges."""
        mesh = partition_domain(
            domain_origin=(0.0,),
            domain_size=(1.0,),
            n_cells=(4,),
            blocks_per_axis=(2,),
            n_ranks=1,
        )
        all_centers = np.concatenate(
            [np.asarray(b.cell_centers(0)) for b in mesh.elements]
        )
        h = 0.25
        assert all_centers[0] == pytest.approx(0.5 * h)
        assert all_centers[-1] == pytest.approx(1.0 - 0.5 * h)

    def test_round_robin_placement(self):
        mesh = partition_domain(
            domain_origin=(0.0,),
            domain_size=(1.0,),
            n_cells=(8,),
            blocks_per_axis=(4,),
            n_ranks=2,
        )
        assert mesh.placement.owner(ComponentId(0)) == 0
        assert mesh.placement.owner(ComponentId(1)) == 1
        assert mesh.placement.owner(ComponentId(2)) == 0
        assert mesh.placement.owner(ComponentId(3)) == 1

    def test_3d_tiling(self):
        n_cells = (8, 8, 8)
        mesh = partition_domain(
            domain_origin=(0.0, 0.0, 0.0),
            domain_size=(1.0, 1.0, 1.0),
            n_cells=n_cells,
            blocks_per_axis=(2, 2, 2),
            n_ranks=4,
        )
        assert len(mesh.elements) == 8
        covered = np.zeros(n_cells, dtype=int)
        for block in mesh.elements:
            covered[block.index_extent.slices] += 1
        assert np.all(covered == 1)

    def test_cell_spacing_uniform(self):
        mesh = self._make_2d()
        for block in mesh.elements:
            assert block.cell_spacing == (0.125, 0.125)

    def test_returns_array_of_block(self):
        mesh = self._make_2d()
        assert isinstance(mesh, Array)
        assert all(isinstance(b, Block) for b in mesh.elements)


class TestCovers:
    def test_single_block_covers_its_own_extent(self):
        mesh = partition_domain(
            domain_origin=(0.0,),
            domain_size=(1.0,),
            n_cells=(8,),
            blocks_per_axis=(1,),
            n_ranks=1,
        )
        assert covers(mesh, Extent((slice(0, 8),)))

    def test_two_blocks_cover_full_domain(self):
        mesh = partition_domain(
            domain_origin=(0.0,),
            domain_size=(1.0,),
            n_cells=(8,),
            blocks_per_axis=(2,),
            n_ranks=1,
        )
        assert covers(mesh, Extent((slice(0, 8),)))

    def test_gap_in_coverage_detected(self):
        """A mesh missing the middle rows does not cover the full extent."""
        from cosmic_foundry.record import Placement

        blocks = (
            Block(
                index_extent=Extent((slice(0, 3),)),
                origin=(0.5,),
                cell_spacing=(1.0,),
            ),
            Block(
                index_extent=Extent((slice(5, 8),)),
                origin=(5.5,),
                cell_spacing=(1.0,),
            ),
        )
        mesh: Array[Block] = Array(
            elements=blocks,
            placement=Placement({ComponentId(0): 0, ComponentId(1): 0}),
        )
        assert not covers(mesh, Extent((slice(0, 8),)))
