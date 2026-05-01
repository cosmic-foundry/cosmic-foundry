"""Algorithm structure contracts for concrete geometry selection."""

from __future__ import annotations

from cosmic_foundry.computation.algorithm_capabilities import (
    AlgorithmCapability,
    AlgorithmRegistry,
    AlgorithmRequest,
    AlgorithmStructureContract,
)

GeometryCapability = AlgorithmCapability
GeometryRegistry = AlgorithmRegistry
GeometryRequest = AlgorithmRequest


def _contract(
    *,
    requires: tuple[str, ...],
    provides: tuple[str, ...],
) -> AlgorithmStructureContract:
    return AlgorithmStructureContract(frozenset(requires), frozenset(provides))


_CAPABILITIES: tuple[GeometryCapability, ...] = (
    GeometryCapability(
        "euclidean_manifold",
        "EuclideanManifold",
        "manifold",
        _contract(
            requires=("dimension",),
            provides=("manifold", "riemannian", "flat_metric", "cartesian_chart"),
        ),
    ),
    GeometryCapability(
        "schwarzschild_manifold",
        "SchwarzschildManifold",
        "manifold",
        _contract(
            requires=("central_mass_symbol",),
            provides=(
                "manifold",
                "pseudo_riemannian",
                "lorentzian_metric",
                "schwarzschild_geometry",
            ),
        ),
    ),
    GeometryCapability(
        "cartesian_mesh",
        "CartesianMesh",
        "mesh",
        _contract(
            requires=("euclidean_manifold", "origin", "spacing", "shape"),
            provides=("mesh", "structured_mesh", "cartesian_mesh", "uniform_grid"),
        ),
    ),
    GeometryCapability(
        "cartesian_exterior_derivative",
        "CartesianExteriorDerivative",
        "discrete_geometry_operator",
        _contract(
            requires=("cartesian_mesh", "discrete_field", "form_degree"),
            provides=(
                "discrete_exterior_derivative",
                "chain_map",
                "exact_stokes",
                "cartesian_mesh",
            ),
        ),
    ),
    GeometryCapability(
        "cartesian_point_restriction",
        "CartesianPointRestriction",
        "restriction_operator",
        _contract(
            requires=("cartesian_mesh", "zero_form"),
            provides=("restriction", "degree_0", "point_field", "cartesian_mesh"),
        ),
    ),
    GeometryCapability(
        "cartesian_edge_restriction",
        "CartesianEdgeRestriction",
        "restriction_operator",
        _contract(
            requires=("cartesian_mesh", "one_form"),
            provides=("restriction", "degree_1", "edge_field", "cartesian_mesh"),
        ),
    ),
    GeometryCapability(
        "cartesian_face_restriction",
        "CartesianFaceRestriction",
        "restriction_operator",
        _contract(
            requires=("cartesian_mesh", "differential_form"),
            provides=("restriction", "face_field", "cartesian_mesh"),
        ),
    ),
    GeometryCapability(
        "cartesian_volume_restriction",
        "CartesianVolumeRestriction",
        "restriction_operator",
        _contract(
            requires=("cartesian_mesh", "zero_form"),
            provides=("restriction", "volume_field", "finite_volume", "cartesian_mesh"),
        ),
    ),
)


GEOMETRY_REGISTRY = GeometryRegistry(_CAPABILITIES)


def geometry_capabilities() -> tuple[GeometryCapability, ...]:
    """Return declared concrete-geometry capabilities."""
    return GEOMETRY_REGISTRY.capabilities


def select_geometry(request: GeometryRequest) -> GeometryCapability:
    """Select a concrete-geometry implementation declaration by capability."""
    return GEOMETRY_REGISTRY.select(request)


__all__ = [
    "GEOMETRY_REGISTRY",
    "GeometryCapability",
    "geometry_capabilities",
    "GeometryRegistry",
    "GeometryRequest",
    "select_geometry",
]
