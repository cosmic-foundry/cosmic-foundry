"""Tests for discretize: (ContinuousField, Array[Patch]) → Array[PatchFunction]."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from cosmic_foundry.field import ContinuousField, PatchFunction
from cosmic_foundry.mesh import discretize, partition_domain
from cosmic_foundry.record import Array, ComponentId


def _mesh_1d(n_cells: int, n_blocks: int, n_ranks: int = 1) -> Array:
    return partition_domain(
        domain_origin=(0.0,),
        domain_size=(1.0,),
        n_cells=(n_cells,),
        blocks_per_axis=(n_blocks,),
        n_ranks=n_ranks,
    )


def _mesh_2d(
    n_cells: tuple[int, int],
    blocks_per_axis: tuple[int, int],
    n_ranks: int = 1,
) -> Array:
    return partition_domain(
        domain_origin=(0.0, 0.0),
        domain_size=(1.0, 1.0),
        n_cells=n_cells,
        blocks_per_axis=blocks_per_axis,
        n_ranks=n_ranks,
    )


class TestDiscretizeOperator:
    """Verify f_h(x_i) = f(x_i) — the mathematical claim of the map."""

    def test_zero_function_produces_zero_payload(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: jnp.zeros_like(x))
        field = discretize(f, _mesh_1d(8, 2))
        for df in field.elements:
            assert jnp.all(df.payload == 0.0)

    def test_constant_function_produces_constant_payload(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: jnp.full_like(x, 42.0))
        field = discretize(f, _mesh_1d(8, 2))
        for df in field.elements:
            assert jnp.all(df.payload == pytest.approx(42.0))

    def test_identity_function_produces_node_positions_1d(self) -> None:
        mesh = _mesh_1d(8, 1)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = discretize(f, mesh)
        assert jnp.allclose(
            field[ComponentId(0)].payload, mesh[ComponentId(0)].node_positions(0)
        )

    def test_multiblock_each_segment_samples_its_own_block(self) -> None:
        mesh = _mesh_1d(8, 2)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = discretize(f, mesh)
        for i, block in enumerate(mesh.elements):
            cid = ComponentId(i)
            assert jnp.allclose(field[cid].payload, block.node_positions(0))

    def test_2d_function_evaluated_at_cell_center_meshgrid(self) -> None:
        mesh = _mesh_2d((4, 6), (1, 1))
        f = ContinuousField(name="phi", fn=lambda x, y: x + y)
        field = discretize(f, mesh)
        block = mesh[ComponentId(0)]
        xs = block.node_positions(0)
        ys = block.node_positions(1)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        assert jnp.allclose(field[ComponentId(0)].payload, X + Y)


class TestDiscretizeCodomain:
    """Verify outputs land in the claimed codomain."""

    def test_payload_shape_matches_block_interior(self) -> None:
        mesh = _mesh_1d(8, 2)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = discretize(f, mesh)
        for i, block in enumerate(mesh.elements):
            assert field[ComponentId(i)].payload.shape == block.shape

    def test_payload_dtype_is_float64(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = discretize(f, _mesh_1d(8, 2))
        for df in field.elements:
            assert df.payload.dtype == jnp.float64

    def test_payload_is_finite(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = discretize(f, _mesh_1d(8, 2))
        for df in field.elements:
            assert jnp.all(jnp.isfinite(df.payload))

    def test_result_is_array_of_discrete_field(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = discretize(f, _mesh_1d(8, 1))
        assert isinstance(field, Array)
        assert all(isinstance(df, PatchFunction) for df in field.elements)


class TestDiscretizeIdentity:
    def test_element_count_matches_block_count(self) -> None:
        mesh = _mesh_2d((8, 8), (2, 2))
        f = ContinuousField(name="phi", fn=lambda x, y: x)
        field = discretize(f, mesh)
        assert len(field.elements) == len(mesh.elements)

    def test_field_name_inherited_from_continuous_field(self) -> None:
        f = ContinuousField(name="density", fn=lambda x: x)
        field = discretize(f, _mesh_1d(8, 1))
        assert all(df.name == "density" for df in field.elements)

    def test_placement_mirrors_mesh_placement(self) -> None:
        mesh = _mesh_1d(8, 4, n_ranks=2)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = discretize(f, mesh)
        assert field.placement == mesh.placement
