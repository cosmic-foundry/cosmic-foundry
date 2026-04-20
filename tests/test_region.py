"""Tests for Region."""

from __future__ import annotations

import pytest

from cosmic_foundry.theory.euclidean_space import EuclideanSpace
from cosmic_foundry.theory.manifold_with_boundary import ManifoldWithBoundary
from cosmic_foundry.theory.region import Region
from cosmic_foundry.theory.smooth_manifold import SmoothManifold


class _Face(ManifoldWithBoundary):
    def __init__(self, ndim: int) -> None:
        self._ndim = ndim

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def boundary(self) -> tuple[ManifoldWithBoundary, ...]:
        return ()


class _Cube(Region):
    """Unit cube in ℝ³."""

    @property
    def ambient_manifold(self) -> SmoothManifold:
        return EuclideanSpace(3)

    @property
    def boundary(self) -> tuple[ManifoldWithBoundary, ...]:
        return tuple(_Face(2) for _ in range(6))


class _Interval(Region):
    """Closed interval in ℝ¹."""

    @property
    def ambient_manifold(self) -> SmoothManifold:
        return EuclideanSpace(1)

    @property
    def boundary(self) -> tuple[ManifoldWithBoundary, ...]:
        return (_Face(0), _Face(0))


def test_region_is_manifold_with_boundary() -> None:
    assert issubclass(Region, ManifoldWithBoundary)


def test_region_is_abstract() -> None:
    with pytest.raises(TypeError):
        Region()  # type: ignore[abstract]


def test_cube_ndim_derived_from_ambient() -> None:
    cube = _Cube()
    assert cube.ndim == 3
    assert cube.ndim == cube.ambient_manifold.ndim


def test_interval_ndim_derived_from_ambient() -> None:
    interval = _Interval()
    assert interval.ndim == 1


def test_ambient_manifold_is_smooth() -> None:
    cube = _Cube()
    assert isinstance(cube.ambient_manifold, SmoothManifold)


def test_boundary_faces_are_codimension_one() -> None:
    cube = _Cube()
    for face in cube.boundary:
        assert face.ndim == cube.ndim - 1


def test_interval_boundary_count() -> None:
    interval = _Interval()
    assert len(interval.boundary) == 2
