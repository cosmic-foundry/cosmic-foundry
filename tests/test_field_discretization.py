"""Tests for ContinuousField.discretize: f.discretize(mesh) → Array[T]."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from cosmic_foundry.computation.array import Array
from cosmic_foundry.mesh import partition_domain
from cosmic_foundry.theory.field import ContinuousField


def _mesh_1d(n_cells: int, n_blocks: int) -> Array:
    return partition_domain(
        domain_origin=(0.0,),
        domain_size=(1.0,),
        n_cells=(n_cells,),
        blocks_per_axis=(n_blocks,),
    )


def _mesh_2d(
    n_cells: tuple[int, int],
    blocks_per_axis: tuple[int, int],
) -> Array:
    return partition_domain(
        domain_origin=(0.0, 0.0),
        domain_size=(1.0, 1.0),
        n_cells=n_cells,
        blocks_per_axis=blocks_per_axis,
    )


class TestDiscretizeOperator:
    """Verify f_h(x_i) = f(x_i) — the mathematical claim of the map."""

    def test_zero_function_produces_zero_payload(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: jnp.zeros_like(x))
        field = f.discretize(_mesh_1d(8, 2))
        for arr in field.elements:
            assert jnp.all(arr == 0.0)

    def test_constant_function_produces_constant_payload(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: jnp.full_like(x, 42.0))
        field = f.discretize(_mesh_1d(8, 2))
        for arr in field.elements:
            assert jnp.all(arr == pytest.approx(42.0))

    def test_identity_function_produces_node_positions_1d(self) -> None:
        mesh = _mesh_1d(8, 1)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = f.discretize(mesh)
        assert jnp.allclose(field[0], mesh[0].node_positions(0))

    def test_multiblock_each_segment_samples_its_own_block(self) -> None:
        mesh = _mesh_1d(8, 2)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = f.discretize(mesh)
        for i, block in enumerate(mesh.elements):
            assert jnp.allclose(field[i], block.node_positions(0))

    def test_2d_function_evaluated_at_cell_center_meshgrid(self) -> None:
        mesh = _mesh_2d((4, 6), (1, 1))
        f = ContinuousField(name="phi", fn=lambda x, y: x + y)
        field = f.discretize(mesh)
        block = mesh[0]
        xs = block.node_positions(0)
        ys = block.node_positions(1)
        X, Y = jnp.meshgrid(xs, ys, indexing="ij")
        assert jnp.allclose(field[0], X + Y)


class TestDiscretizeCodomain:
    """Verify outputs land in the claimed codomain."""

    def test_payload_shape_matches_block_interior(self) -> None:
        mesh = _mesh_1d(8, 2)
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = f.discretize(mesh)
        for i, block in enumerate(mesh.elements):
            assert field[i].shape == block.shape

    def test_payload_dtype_is_float64(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = f.discretize(_mesh_1d(8, 2))
        for arr in field.elements:
            assert arr.dtype == jnp.float64

    def test_payload_is_finite(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = f.discretize(_mesh_1d(8, 2))
        for arr in field.elements:
            assert jnp.all(jnp.isfinite(arr))

    def test_result_is_array(self) -> None:
        f = ContinuousField(name="phi", fn=lambda x: x)
        field = f.discretize(_mesh_1d(8, 1))
        assert isinstance(field, Array)


class TestDiscretizeIdentity:
    def test_element_count_matches_block_count(self) -> None:
        mesh = _mesh_2d((8, 8), (2, 2))
        f = ContinuousField(name="phi", fn=lambda x, y: x)
        field = f.discretize(mesh)
        assert len(field.elements) == len(mesh.elements)
