"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import pytest

from cosmic_foundry.geometry.domain import Domain
from cosmic_foundry.theory.euclidean_space import EuclideanSpace


@pytest.fixture
def euclidean_domain_3d() -> Domain:
    return Domain(
        manifold=EuclideanSpace(3),
        origin=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
    )
