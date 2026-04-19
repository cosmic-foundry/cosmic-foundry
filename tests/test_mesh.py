"""Tests for the uniform structured mesh: Patch, partition_domain, covers."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_foundry.computation.array import Array
from cosmic_foundry.computation.descriptor import Extent
from cosmic_foundry.mesh import Patch, covers, partition_domain


class TestPatch:
    def _make_patch(self) -> Patch:
        return Patch(
            index_extent=Extent((slice(0, 8), slice(0, 8), slice(0, 8))),
            origin=(0.0625, 0.0625, 0.0625),
            cell_spacing=(0.125, 0.125, 0.125),
        )

    def test_shape_and_ndim(self):
        p = self._make_patch()
        assert p.shape == (8, 8, 8)
        assert p.ndim == 3

    def test_node_positions_values(self):
        h = 0.25
        patch = Patch(
            index_extent=Extent((slice(0, 4),)),
            origin=(0.5 * h,),
            cell_spacing=(h,),
        )
        positions = np.asarray(patch.node_positions(0))
        expected = np.array([0.125, 0.375, 0.625, 0.875])
        np.testing.assert_allclose(positions, expected)

    def test_node_positions_spacing(self):
        p = self._make_patch()
        positions = np.asarray(p.node_positions(0))
        diffs = np.diff(positions)
        np.testing.assert_allclose(diffs, p.cell_spacing[0])


class TestPartitionDomain:
    def _make_2d(self, **kwargs) -> Array[Patch]:
        defaults = dict(
            domain_origin=(0.0, 0.0),
            domain_size=(1.0, 1.0),
            n_cells=(8, 8),
            blocks_per_axis=(2, 2),
        )
        defaults.update(kwargs)
        return partition_domain.execute(**defaults)

    def test_patch_count(self):
        mesh = self._make_2d()
        assert len(mesh.elements) == 4  # 2×2

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            partition_domain.execute(
                domain_origin=(0.0,),
                domain_size=(1.0,),
                n_cells=(7,),
                blocks_per_axis=(2,),
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            partition_domain.execute(
                domain_origin=(0.0,),
                domain_size=(1.0, 1.0),
                n_cells=(8,),
                blocks_per_axis=(2,),
            )

    def test_index_extents_tile_domain(self):
        n_cells = (8, 12)
        mesh = partition_domain.execute(
            domain_origin=(0.0, 0.0),
            domain_size=(1.0, 1.0),
            n_cells=n_cells,
            blocks_per_axis=(2, 3),
        )
        covered = np.zeros(n_cells, dtype=int)
        for patch in mesh.elements:
            covered[patch.index_extent.slices] += 1
        assert np.all(covered == 1), "patches must tile the domain exactly once"

    def test_node_positions_cover_domain(self):
        """First and last node positions sit h/2 from the domain edges."""
        mesh = partition_domain.execute(
            domain_origin=(0.0,),
            domain_size=(1.0,),
            n_cells=(4,),
            blocks_per_axis=(2,),
        )
        all_positions = np.concatenate(
            [np.asarray(p.node_positions(0)) for p in mesh.elements]
        )
        h = 0.25
        assert all_positions[0] == pytest.approx(0.5 * h)
        assert all_positions[-1] == pytest.approx(1.0 - 0.5 * h)

    def test_3d_tiling(self):
        n_cells = (8, 8, 8)
        mesh = partition_domain.execute(
            domain_origin=(0.0, 0.0, 0.0),
            domain_size=(1.0, 1.0, 1.0),
            n_cells=n_cells,
            blocks_per_axis=(2, 2, 2),
        )
        assert len(mesh.elements) == 8
        covered = np.zeros(n_cells, dtype=int)
        for patch in mesh.elements:
            covered[patch.index_extent.slices] += 1
        assert np.all(covered == 1)

    def test_cell_spacing_uniform(self):
        mesh = self._make_2d()
        for patch in mesh.elements:
            assert patch.cell_spacing == (0.125, 0.125)

    def test_returns_array_of_patch(self):
        mesh = self._make_2d()
        assert isinstance(mesh, Array)
        assert all(isinstance(p, Patch) for p in mesh.elements)


class TestCovers:
    def test_single_patch_covers_its_own_extent(self):
        mesh = partition_domain.execute(
            domain_origin=(0.0,),
            domain_size=(1.0,),
            n_cells=(8,),
            blocks_per_axis=(1,),
        )
        assert covers(mesh, Extent((slice(0, 8),)))

    def test_two_patches_cover_full_domain(self):
        mesh = partition_domain.execute(
            domain_origin=(0.0,),
            domain_size=(1.0,),
            n_cells=(8,),
            blocks_per_axis=(2,),
        )
        assert covers(mesh, Extent((slice(0, 8),)))

    def test_gap_in_coverage_detected(self):
        """A mesh missing the middle rows does not cover the full extent."""
        patches = (
            Patch(
                index_extent=Extent((slice(0, 3),)),
                origin=(0.5,),
                cell_spacing=(1.0,),
            ),
            Patch(
                index_extent=Extent((slice(5, 8),)),
                origin=(5.5,),
                cell_spacing=(1.0,),
            ),
        )
        mesh: Array[Patch] = Array(elements=patches)
        assert not covers(mesh, Extent((slice(0, 8),)))
