"""Tests for FieldDiscretization: (ContinuousField, G) → DiscreteField."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from cosmic_foundry.fields import ContinuousField
from cosmic_foundry.mesh import (
    DistributedField,
    FieldDiscretization,
    UniformGrid,
    partition_domain,
)


def _grid_1d(n_cells: int, n_blocks: int, n_ranks: int = 1) -> UniformGrid:
    return partition_domain(
        domain_origin=(0.0,),
        domain_size=(1.0,),
        n_cells=(n_cells,),
        blocks_per_axis=(n_blocks,),
        n_ranks=n_ranks,
    )


def _grid_2d(
    n_cells: tuple[int, int],
    blocks_per_axis: tuple[int, int],
    n_ranks: int = 1,
) -> UniformGrid:
    return partition_domain(
        domain_origin=(0.0, 0.0),
        domain_size=(1.0, 1.0),
        n_cells=n_cells,
        blocks_per_axis=blocks_per_axis,
        n_ranks=n_ranks,
    )


class TestFieldDiscretizationOperator:
    """Verify f_h(x_i) = f(x_i) — the mathematical claim of the map."""

    def test_zero_function_produces_zero_payload(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: jnp.zeros_like(x))
        field = FieldDiscretization().execute(f, _grid_1d(8, 2))
        for seg in field.segments:
            assert jnp.all(seg.payload == 0.0)

    def test_constant_function_produces_constant_payload(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: jnp.full_like(x, 42.0))
        field = FieldDiscretization().execute(f, _grid_1d(8, 2))
        for seg in field.segments:
            assert jnp.all(seg.payload == pytest.approx(42.0))

    def test_identity_function_produces_cell_centers_1d(self) -> None:
        grid = _grid_1d(8, 1)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, grid)
        assert jnp.allclose(field.segments[0].payload, grid.blocks[0].cell_centers(0))

    def test_multiblock_each_segment_samples_its_own_block(self) -> None:
        grid = _grid_1d(8, 2)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, grid)
        for block in grid.blocks:
            seg = field.segment(block.block_id)
            assert jnp.allclose(seg.payload, block.cell_centers(0))

    def test_2d_function_evaluated_at_cell_center_meshgrid(self) -> None:
        grid = _grid_2d((4, 6), (1, 1))
        f = ContinuousField(name="phi", fn=lambda x, y: x + y)
        field = FieldDiscretization().execute(f, grid)
        block = grid.blocks[0]
        xs = block.cell_centers(0)
        ys = block.cell_centers(1)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        assert jnp.allclose(field.segments[0].payload, X + Y)


class TestFieldDiscretizationCodomain:
    """Verify outputs land in the claimed codomain."""

    def test_payload_shape_matches_block_interior(self) -> None:
        grid = _grid_1d(8, 2)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, grid)
        for block in grid.blocks:
            seg = field.segment(block.block_id)
            assert seg.payload.shape == block.shape

    def test_payload_dtype_is_float64(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, _grid_1d(8, 2))
        for seg in field.segments:
            assert seg.payload.dtype == jnp.float64

    def test_payload_is_finite(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, _grid_1d(8, 2))
        for seg in field.segments:
            assert jnp.all(jnp.isfinite(seg.payload))

    def test_extent_equals_block_index_extent(self) -> None:
        grid = _grid_1d(8, 2)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, grid)
        for block in grid.blocks:
            seg = field.segment(block.block_id)
            assert seg.extent == block.index_extent

    def test_no_ghost_cells(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, _grid_1d(8, 2))
        for seg in field.segments:
            assert seg.interior_extent is None

    def test_result_is_distributed_field(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, _grid_1d(8, 1))
        assert isinstance(field, DistributedField)


class TestFieldDiscretizationIdentity:
    def test_segment_count_matches_block_count(self) -> None:
        grid = _grid_2d((8, 8), (2, 2))
        f = ContinuousField(name="phi", fn=lambda x, y: x)
        field = FieldDiscretization().execute(f, grid)
        assert len(field.segments) == len(grid.blocks)

    def test_segment_ids_match_block_ids(self) -> None:
        grid = _grid_1d(8, 4)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, grid)
        seg_ids = {seg.segment_id for seg in field.segments}
        block_ids = {b.block_id for b in grid.blocks}
        assert seg_ids == block_ids

    def test_field_name_inherited_from_continuous_field(self) -> None:
        f = ContinuousField(name="density", fn=lambda x: x)
        field = FieldDiscretization().execute(f, _grid_1d(8, 1))
        assert field.name == "density"

    def test_placement_mirrors_grid_rank_map(self) -> None:
        grid = _grid_1d(8, 4, n_ranks=2)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = FieldDiscretization().execute(f, grid)
        for block in grid.blocks:
            assert field.placement.owner(block.block_id) == grid.owner(block.block_id)
