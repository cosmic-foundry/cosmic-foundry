"""Capability-atlas structural helpers.

This module holds the generated-atlas projection machinery used by
``tests.test_structure`` and ``scripts.gen_capability_atlas_docs``.
"""

from __future__ import annotations

import ast
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any, NamedTuple, NewType, TypeAlias

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    ComparisonPredicate,
    CoverageRegion,
    DecompositionField,
    DescriptorCoordinate,
    DescriptorField,
    EvidencePredicate,
    InvalidCellRule,
    LinearSolverField,
    MapStructureField,
    MembershipPredicate,
    NumericInterval,
    ParameterAxis,
    ParameterBin,
    ParameterDescriptor,
    ParameterSpaceSchema,
    ReactionNetworkField,
    SolveRelationField,
    StructuredPredicate,
    decomposition_parameter_schema,
    linear_operator_descriptor_from_assembled_operator,
    linear_operator_descriptor_from_solve_relation_descriptor,
    linear_solver_parameter_schema,
    map_structure_parameter_schema,
    predicate_sets_are_disjoint,
    reaction_network_parameter_schema,
    solve_relation_parameter_schema,
)
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.decompositions.capabilities import (
    decomposition_coverage_regions,
)
from cosmic_foundry.computation.solvers.capabilities import (
    least_squares_solver_coverage_regions,
    linear_solver_coverage_regions,
    root_solver_coverage_regions,
    spectral_solver_coverage_regions,
)
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.capabilities import (
    nordsieck_history_descriptor,
    rhs_evaluation_descriptor,
    rhs_history_descriptor,
    rhs_step_diagnostics_descriptor,
    time_integration_step_map_regions,
)
from cosmic_foundry.computation.time_integrators.constraint_aware import (
    NuclearStatisticalEquilibriumSolver,
    reaction_network_coverage_regions,
)
from cosmic_foundry.computation.time_integrators.domains import NonnegativeStateDomain
from cosmic_foundry.computation.time_integrators.implicit import (
    ImplicitRungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.integrator import ODEState
from tests import test_structure as structure
from tests.claims import Claim

_PROJECT_ROOT = Path(__file__).parent.parent
_SolveRelationSchemaClaim = structure._SolveRelationSchemaClaim
_ATLAS_TIME_BACKEND = NumpyBackend()


def _atlas_linear_operator_descriptor(
    matrix: tuple[tuple[float, ...], ...],
    rhs: tuple[float, ...],
) -> ParameterDescriptor:
    return linear_operator_descriptor_from_assembled_operator(
        structure._MatrixLinearOperator(matrix),
        Tensor(rhs, backend=_ATLAS_TIME_BACKEND),
    )


def _atlas_linear_residual_projection() -> ParameterDescriptor:
    matrix = tuple(
        tuple(2.0 if row == column else 0.0 for column in range(4)) for row in range(4)
    )
    return _atlas_linear_operator_descriptor(matrix, (1.0, 0.0, 0.0, 0.0))


def _atlas_affine_target_zero_projection() -> ParameterDescriptor:
    state = ODEState(0.0, Tensor([1.0, 2.0], backend=_ATLAS_TIME_BACKEND))
    descriptor = ImplicitRungeKuttaIntegrator(1).step_solve_relation_descriptor(
        structure._AffineTestRHS(), state, 0.125
    )
    return linear_operator_descriptor_from_solve_relation_descriptor(descriptor)


_AtlasText = NewType("_AtlasText", str)
_AtlasDescriptorField: TypeAlias = (
    SolveRelationField
    | LinearSolverField
    | DecompositionField
    | ReactionNetworkField
    | MapStructureField
)
_AtlasDescriptorGroup: TypeAlias = tuple[ParameterDescriptor, ...]


class _AtlasAxisView(NamedTuple):
    """Visualization policy derived from a schema axis partition."""

    cells: tuple[ParameterBin | NumericInterval, ...]
    use_log_scale: bool


class _AtlasProjectionObjective(NamedTuple):
    """Lexicographic loss of quotienting schema cells onto a plot plane."""

    hidden_uncovered_cells: int
    negative_visible_uncovered_cells: int
    signature_loss: int
    negative_visible_owned_cells: int
    negative_projected_cells: int


class _AtlasProjection(NamedTuple):
    """Schema-axis quotient selected by semantic information preservation."""

    x_axis: ParameterAxis
    y_axis: ParameterAxis
    objective: _AtlasProjectionObjective


class _AtlasUncoveredCell(NamedTuple):
    """Computed valid schema cell not owned by any coverage claim."""

    predicates: tuple[StructuredPredicate, ...]

    def contains(self, descriptor: ParameterDescriptor) -> bool:
        return all(predicate.evaluate(descriptor) for predicate in self.predicates)


_AtlasRegionSource: TypeAlias = InvalidCellRule | CoverageRegion | _AtlasUncoveredCell


class _AtlasCellSource(NamedTuple):
    """Semantic source identity for one schema cell."""

    kind: type[InvalidCellRule] | type[CoverageRegion] | type[_AtlasUncoveredCell]
    key: tuple[object, ...]


_AtlasCellSignature: TypeAlias = frozenset[_AtlasCellSource]
_ATLAS_PROJECTIONS_BY_SCHEMA: dict[tuple[object, ...], _AtlasProjection] = {}
_ATLAS_CANDIDATE_PROJECTIONS_BY_SCHEMA: dict[
    tuple[object, ...], tuple[_AtlasProjection, ...]
] = {}
_ATLAS_SIGNATURE_CELLS_BY_SCHEMA: dict[
    tuple[object, ...], tuple[tuple[tuple[int, ...], _AtlasCellSignature], ...]
] = {}
_ATLAS_UNCOVERED_CELLS_BY_SCHEMA: dict[
    tuple[object, ...], tuple[_AtlasUncoveredCell, ...]
] = {}
_SCHEMA_INDEXED_CELLS_BY_AXES: dict[
    tuple[object, ...],
    tuple[tuple[tuple[int, ...], tuple[StructuredPredicate, ...]], ...],
] = {}
_AXIS_CELLS_BY_AXIS: dict[
    tuple[object, tuple[ParameterBin | NumericInterval, ...]],
    tuple[tuple[StructuredPredicate, ...], ...],
] = {}
_PREDICATE_SET_DISJOINTNESS: dict[tuple[int, int], bool] = {}
_CELL_PREDICATE_DISJOINTNESS: dict[tuple[tuple[int, ...], int], bool] = {}


class _AtlasAffineDecayRHS:
    def __init__(self, rate: float) -> None:
        self._rate = rate

    def __call__(self, _t: float, u: Tensor) -> Tensor:
        return Tensor([-self._rate * float(u[0])], backend=u.backend)

    def jacobian(self, _t: float, u: Tensor) -> Tensor:
        return Tensor([[-self._rate]], backend=u.backend)


class _AtlasBoundaryApproachRHS:
    state_domain = NonnegativeStateDomain(1)

    def __call__(self, _t: float, u: Tensor) -> Tensor:
        return Tensor([-1.0], backend=u.backend)

    def jacobian(self, _t: float, u: Tensor) -> Tensor:
        return Tensor([[0.0]], backend=u.backend)


def _atlas_scalar_state(value: float = 1.0) -> ODEState:
    return ODEState(0.0, Tensor([value], backend=_ATLAS_TIME_BACKEND))


def _atlas_step_diagnostic_descriptors() -> tuple[ParameterDescriptor, ...]:
    return (
        rhs_step_diagnostics_descriptor(
            _AtlasAffineDecayRHS(0.1),
            _atlas_scalar_state(),
            1.0e-1,
            local_error_target=1.0e-6,
            retry_budget=3,
        ),
        rhs_step_diagnostics_descriptor(
            _AtlasAffineDecayRHS(100.0),
            _atlas_scalar_state(),
            1.0e-1,
        ),
        rhs_step_diagnostics_descriptor(
            _AtlasBoundaryApproachRHS(),
            _atlas_scalar_state(0.1),
            1.0e-1,
        ),
    )


@cache
def _capability_atlas_descriptors() -> tuple[ParameterDescriptor, ...]:
    return (
        _SolveRelationSchemaClaim._solve_descriptor(
            dim_x=3,
            dim_y=5,
            objective_relation="least_squares",
            acceptance_relation="objective_minimum",
        ),
        _SolveRelationSchemaClaim._solve_descriptor(
            map_linearity_defect=None,
            map_linearity_evidence="unavailable",
            residual_target_available=False,
        ),
        _SolveRelationSchemaClaim._solve_descriptor(
            target_is_zero=True,
            map_linearity_defect=1.0,
            matrix_representation_available=False,
            derivative_oracle_kind="none",
        ),
        _SolveRelationSchemaClaim._solve_descriptor(
            target_is_zero=True,
            map_linearity_defect=1.0,
            matrix_representation_available=False,
            derivative_oracle_kind="jvp",
        ),
        _SolveRelationSchemaClaim._solve_descriptor(
            auxiliary_scalar_count=1,
            normalization_constraint_count=1,
            acceptance_relation="eigenpair_residual",
            objective_relation="spectral_residual",
        ),
        _SolveRelationSchemaClaim._solve_descriptor(
            acceptance_relation="eigenpair_residual",
        ),
        _SolveRelationSchemaClaim._implicit_stage_descriptor(),
        _atlas_affine_target_zero_projection(),
        _atlas_linear_residual_projection(),
        _atlas_linear_operator_descriptor(
            ((2.0, -1.0), (-1.0, 2.0)),
            (1.0, 0.0),
        ),
        _atlas_linear_operator_descriptor(
            ((1.0, 1.0), (1.0, 1.0)),
            (1.0, 1.0),
        ),
        _SolveRelationSchemaClaim._linear_descriptor(
            linear_operator_matrix_available=False,
            matrix_representation_available=False,
        ),
        _SolveRelationSchemaClaim._linear_descriptor(dim_y=5),
        _SolveRelationSchemaClaim._decomposition_descriptor(),
        _SolveRelationSchemaClaim._decomposition_descriptor(
            matrix_columns=5,
        ),
        _SolveRelationSchemaClaim._reaction_network_descriptor(),
        rhs_evaluation_descriptor(),
        rhs_history_descriptor(),
        nordsieck_history_descriptor(1.0e-2),
        *_atlas_step_diagnostic_descriptors(),
    )


def _capability_atlas_schemas() -> tuple[ParameterSpaceSchema, ...]:
    return (
        solve_relation_parameter_schema(),
        linear_solver_parameter_schema(),
        decomposition_parameter_schema(),
        reaction_network_parameter_schema(),
        map_structure_parameter_schema(),
    )


def _capability_atlas_coverage_regions() -> tuple[CoverageRegion, ...]:
    return (
        *decomposition_coverage_regions(),
        *linear_solver_coverage_regions(),
        *least_squares_solver_coverage_regions(),
        *root_solver_coverage_regions(),
        *spectral_solver_coverage_regions(),
        *reaction_network_coverage_regions(),
        *time_integration_step_map_regions(),
    )


def _atlas_schema_for_descriptor(
    descriptor: ParameterDescriptor,
) -> ParameterSpaceSchema:
    candidates = tuple(
        schema
        for schema in _capability_atlas_schemas()
        if _descriptor_inhabits_schema(descriptor, schema)
    )
    minimal = tuple(
        schema
        for schema in candidates
        if not any(
            other.descriptor_fields < schema.descriptor_fields for other in candidates
        )
    )
    if len(minimal) != 1:
        raise AssertionError(
            "atlas descriptor inhabits "
            f"{len(minimal)} minimal schemas: {[schema.name for schema in minimal]}"
        )
    return minimal[0]


def _descriptor_inhabits_schema(
    descriptor: ParameterDescriptor,
    schema: ParameterSpaceSchema,
) -> bool:
    try:
        schema.validate_descriptor(descriptor)
    except ValueError:
        return False
    return True


def _capability_atlas_descriptor_groups() -> tuple[_AtlasDescriptorGroup, ...]:
    groups: dict[frozenset[DescriptorField], list[ParameterDescriptor]] = {}
    for descriptor in _capability_atlas_descriptors():
        schema = _atlas_schema_for_descriptor(descriptor)
        groups.setdefault(schema.descriptor_fields, []).append(descriptor)
    return tuple(tuple(group) for group in groups.values())


def _atlas_group_schema(group: _AtlasDescriptorGroup) -> ParameterSpaceSchema:
    """Common schema for the descriptors shown by one atlas plot."""
    schema = _atlas_schema_for_descriptor(group[0])
    assert all(
        _atlas_schema_for_descriptor(descriptor) == schema for descriptor in group
    )
    return schema


def _atlas_group_x_axis(group: _AtlasDescriptorGroup) -> _AtlasDescriptorField:
    """X axis selected by schema-cell information preservation."""
    return _atlas_axis_field(_atlas_group_projection(group).x_axis)


def _atlas_group_y_axis(group: _AtlasDescriptorGroup) -> _AtlasDescriptorField:
    """Y axis selected by schema-cell information preservation."""
    return _atlas_axis_field(_atlas_group_projection(group).y_axis)


def _atlas_group_projection(group: _AtlasDescriptorGroup) -> _AtlasProjection:
    schema = _atlas_group_schema(group)
    regions = _atlas_regions_for_schema(schema)
    key = _atlas_schema_region_key(schema, regions)
    if key not in _ATLAS_PROJECTIONS_BY_SCHEMA:
        _ATLAS_PROJECTIONS_BY_SCHEMA[key] = min(
            _atlas_candidate_projections(schema, regions),
            key=lambda item: item.objective,
        )
    return _ATLAS_PROJECTIONS_BY_SCHEMA[key]


def _atlas_schema_region_key(
    schema: ParameterSpaceSchema,
    regions: tuple[CoverageRegion, ...],
) -> tuple[object, ...]:
    return (
        tuple((axis.field, axis.bins) for axis in schema.axes),
        tuple(
            tuple(_predicate_key(predicate) for predicate in rule.predicates)
            for rule in schema.invalid_cells
        ),
        tuple(
            (
                region.owner,
                tuple(_predicate_key(predicate) for predicate in region.predicates),
            )
            for region in regions
        ),
    )


def _atlas_candidate_projections(
    schema: ParameterSpaceSchema,
    regions: tuple[CoverageRegion, ...],
) -> tuple[_AtlasProjection, ...]:
    key = _atlas_schema_region_key(schema, regions)
    if key not in _ATLAS_CANDIDATE_PROJECTIONS_BY_SCHEMA:
        _ATLAS_CANDIDATE_PROJECTIONS_BY_SCHEMA[key] = (
            _compute_atlas_candidate_projections(schema, regions)
        )
    return _ATLAS_CANDIDATE_PROJECTIONS_BY_SCHEMA[key]


def _compute_atlas_candidate_projections(
    schema: ParameterSpaceSchema,
    regions: tuple[CoverageRegion, ...],
) -> tuple[_AtlasProjection, ...]:
    candidates: list[_AtlasProjection] = []
    signature_cells = _schema_signature_cells(schema, regions)
    for x_index, x_axis in enumerate(schema.axes):
        for y_axis in schema.axes[x_index + 1 :]:
            candidates.append(
                _AtlasProjection(
                    x_axis,
                    y_axis,
                    _atlas_projection_objective(
                        schema, signature_cells, x_axis, y_axis
                    ),
                )
            )
    return tuple(candidates)


def _atlas_projection_objective(
    schema: ParameterSpaceSchema,
    signature_cells: tuple[tuple[tuple[int, ...], _AtlasCellSignature], ...],
    x_axis: ParameterAxis,
    y_axis: ParameterAxis,
) -> _AtlasProjectionObjective:
    projected = _atlas_projection_fibers(schema, signature_cells, x_axis, y_axis)
    return _AtlasProjectionObjective(
        hidden_uncovered_cells=_atlas_projection_hidden_uncovered_cells(projected),
        negative_visible_uncovered_cells=-_atlas_projection_visible_uncovered_cells(
            projected
        ),
        signature_loss=_atlas_projection_signature_loss(projected),
        negative_visible_owned_cells=-_atlas_projection_visible_owned_cells(projected),
        negative_projected_cells=-len(projected),
    )


def _atlas_projection_fibers(
    schema: ParameterSpaceSchema,
    signature_cells: tuple[tuple[tuple[int, ...], _AtlasCellSignature], ...],
    x_axis: ParameterAxis,
    y_axis: ParameterAxis,
) -> dict[tuple[int, int], set[_AtlasCellSignature]]:
    projected: dict[tuple[int, int], set[_AtlasCellSignature]] = {}
    x_index = schema.axes.index(x_axis)
    y_index = schema.axes.index(y_axis)
    for coordinates, signature in signature_cells:
        bucket = (
            coordinates[x_index],
            coordinates[y_index],
        )
        projected.setdefault(bucket, set()).add(signature)
    return projected


def _atlas_projection_signature_loss(
    projected: dict[tuple[int, int], set[_AtlasCellSignature]],
) -> int:
    return sum(len(signatures) - 1 for signatures in projected.values())


def _atlas_projection_hidden_uncovered_cells(
    projected: dict[tuple[int, int], set[_AtlasCellSignature]],
) -> int:
    return sum(
        any(
            _atlas_signature_has_source(signature, _AtlasUncoveredCell)
            for signature in signatures
        )
        and any(
            _atlas_signature_has_source(signature, CoverageRegion)
            for signature in signatures
        )
        for signatures in projected.values()
    )


def _atlas_projection_visible_uncovered_cells(
    projected: dict[tuple[int, int], set[_AtlasCellSignature]],
) -> int:
    return sum(
        any(
            _atlas_signature_has_source(signature, _AtlasUncoveredCell)
            for signature in signatures
        )
        and not any(
            _atlas_signature_has_source(signature, CoverageRegion)
            for signature in signatures
        )
        for signatures in projected.values()
    )


def _atlas_projection_visible_owned_cells(
    projected: dict[tuple[int, int], set[_AtlasCellSignature]],
) -> int:
    return sum(
        any(
            _atlas_signature_has_source(signature, CoverageRegion)
            for signature in signatures
        )
        for signatures in projected.values()
    )


def _atlas_signature_has_source(
    signature: _AtlasCellSignature,
    kind: type[InvalidCellRule] | type[CoverageRegion] | type[_AtlasUncoveredCell],
) -> bool:
    return any(source.kind is kind for source in signature)


def _schema_signature_cells(
    schema: ParameterSpaceSchema,
    regions: tuple[CoverageRegion, ...],
) -> tuple[tuple[tuple[int, ...], _AtlasCellSignature], ...]:
    key = _atlas_schema_region_key(schema, regions)
    if key not in _ATLAS_SIGNATURE_CELLS_BY_SCHEMA:
        _ATLAS_SIGNATURE_CELLS_BY_SCHEMA[key] = tuple(
            (coordinates, _schema_cell_signature(schema, regions, coordinates, cell))
            for coordinates, cell in _schema_indexed_cells(schema)
        )
    return _ATLAS_SIGNATURE_CELLS_BY_SCHEMA[key]


def _schema_cell_signature(
    schema: ParameterSpaceSchema,
    regions: tuple[CoverageRegion, ...],
    coordinates: tuple[int, ...],
    cell: tuple[StructuredPredicate, ...],
) -> _AtlasCellSignature:
    sources: set[_AtlasCellSource] = set()
    overlapping_sources = tuple(
        source
        for source in (*schema.invalid_cells, *regions)
        if not _cell_is_provably_disjoint_from_source(cell, source.predicates)
    )
    sources.update(_atlas_cell_source(source) for source in overlapping_sources)
    if not _cell_has_covering_source(cell, overlapping_sources):
        sources.add(_atlas_uncovered_cell_source(coordinates))
    return frozenset(sources)


def _atlas_group_x_range(group: _AtlasDescriptorGroup) -> tuple[float, float]:
    """Visual range derived from the projected schema axis partition."""
    axis = _atlas_schema_axis(_atlas_group_schema(group), _atlas_group_x_axis(group))
    return _atlas_axis_range(axis)


def _atlas_group_y_range(group: _AtlasDescriptorGroup) -> tuple[float, float]:
    """Visual range derived from the projected schema axis partition."""
    axis = _atlas_schema_axis(_atlas_group_schema(group), _atlas_group_y_axis(group))
    return _atlas_axis_range(axis)


def _atlas_group_filename(group: _AtlasDescriptorGroup) -> _AtlasText:
    """Human filename for the generated plot."""
    return _AtlasText(f"{_atlas_group_schema(group).name}.svg")


def _atlas_group_title(group: _AtlasDescriptorGroup) -> _AtlasText:
    """Human title for the generated plot."""
    schema = _atlas_group_schema(group)
    return _AtlasText(f"{schema.name.replace('_', '-').title()} Regions")


def _atlas_group_caption(group: _AtlasDescriptorGroup) -> _AtlasText:
    """Human caption for the generated plot."""
    return _AtlasText(
        f"{_atlas_group_title(group)} over {_field_label(_atlas_group_x_axis(group))} "
        f"and {_field_label(_atlas_group_y_axis(group))}."
    )


def _atlas_axis_view(axis: ParameterAxis) -> _AtlasAxisView:
    return _AtlasAxisView(
        cells=axis.bins,
        use_log_scale=_axis_uses_log_scale(axis),
    )


def _axis_uses_log_scale(axis: ParameterAxis) -> bool:
    numeric_cells = tuple(
        cell for cell in axis.bins if isinstance(cell, NumericInterval)
    )
    return bool(numeric_cells) and all(
        cell.lower is not None
        and (cell.lower > 0.0 or (cell.lower == 0.0 and not cell.include_lower))
        and (cell.upper is None or cell.upper > 0.0)
        for cell in numeric_cells
    )


def _atlas_schema_axis(
    schema: ParameterSpaceSchema,
    field: _AtlasDescriptorField,
) -> ParameterAxis:
    return next(axis for axis in schema.axes if axis.field == field)


def _atlas_axis_range(axis: ParameterAxis) -> tuple[float, float]:
    view = _atlas_axis_view(axis)
    return (0.0, float(len(view.cells)))


def _atlas_axis_coordinate(
    axis: ParameterAxis,
    coordinate: DescriptorCoordinate,
) -> float:
    assert coordinate.known
    return next(
        index + 0.5
        for index, cell in enumerate(_atlas_axis_view(axis).cells)
        if coordinate.value is not None and cell.contains(coordinate.value)
    )


def _atlas_axis_field(axis: ParameterAxis) -> _AtlasDescriptorField:
    assert isinstance(
        axis.field,
        SolveRelationField
        | LinearSolverField
        | DecompositionField
        | ReactionNetworkField
        | MapStructureField,
    )
    return axis.field


def _atlas_fixed_axes(
    group: _AtlasDescriptorGroup,
) -> tuple[_AtlasDescriptorField, ...]:
    """Return non-plotted schema axes fixed to one known value across the plot."""
    return tuple(
        field
        for field in _atlas_hidden_axes(group)
        if _axis_has_one_known_value(group, field)
    )


def _atlas_marginalized_axes(
    group: _AtlasDescriptorGroup,
) -> tuple[_AtlasDescriptorField, ...]:
    """Return non-plotted schema axes not fixed by the descriptor evidence."""
    fixed = set(_atlas_fixed_axes(group))
    return tuple(field for field in _atlas_hidden_axes(group) if field not in fixed)


def _atlas_hidden_axes(
    group: _AtlasDescriptorGroup,
) -> tuple[_AtlasDescriptorField, ...]:
    shown = {_atlas_group_x_axis(group), _atlas_group_y_axis(group)}
    return tuple(
        field
        for axis in _atlas_group_schema(group).axes
        if (field := _atlas_axis_field(axis)) not in shown
    )


def _axis_has_one_known_value(
    descriptors: tuple[ParameterDescriptor, ...],
    field: _AtlasDescriptorField,
) -> bool:
    coordinates = tuple(descriptor.coordinate(field) for descriptor in descriptors)
    return (
        all(coordinate.known for coordinate in coordinates)
        and len({coordinate.value for coordinate in coordinates}) == 1
    )


def _descriptor_value(descriptor: ParameterDescriptor, field: DescriptorField) -> str:
    coordinate = descriptor.coordinate(field)
    value = "unknown" if coordinate.value is None else str(coordinate.value)
    if coordinate.evidence != "exact":
        return f"{value} ({coordinate.evidence})"
    return value


def _field_label(field: Any) -> str:
    if isinstance(
        field,
        SolveRelationField
        | LinearSolverField
        | DecompositionField
        | ReactionNetworkField,
    ):
        return str(field.value)
    return str(field)


def _predicate_label(predicate: Any) -> str:
    if isinstance(predicate, ComparisonPredicate):
        return f"{_field_label(predicate.field)} {predicate.operator} {predicate.value}"
    if isinstance(predicate, AffineComparisonPredicate):
        terms = " + ".join(
            f"{coefficient:g}*{_field_label(field)}"
            for field, coefficient in sorted(
                predicate.terms.items(), key=lambda item: _field_label(item[0])
            )
        )
        if predicate.offset:
            terms = f"{terms} + {predicate.offset:g}"
        return f"{terms} {predicate.operator} {predicate.value:g}"
    if isinstance(predicate, MembershipPredicate):
        values = ", ".join(str(value) for value in sorted(predicate.values, key=str))
        return f"{_field_label(predicate.field)} in {{{values}}}"
    if isinstance(predicate, EvidencePredicate):
        evidence = ", ".join(sorted(predicate.evidence))
        return f"{_field_label(predicate.field)} evidence in {{{evidence}}}"
    return repr(predicate)


def _source_alternatives(
    schema: ParameterSpaceSchema,
    source: _AtlasRegionSource,
    regions: tuple[CoverageRegion, ...] = (),
) -> tuple[tuple[Any, ...], ...]:
    if isinstance(source, _AtlasUncoveredCell):
        return (source.predicates,)
    if isinstance(source, InvalidCellRule):
        assert source in schema.invalid_cells
        return (source.predicates,)
    assert isinstance(source, CoverageRegion)
    assert source in regions
    schema.validate_coverage_region(source)
    return (source.predicates,)


def _cell_has_covering_source(
    cell: tuple[StructuredPredicate, ...],
    sources: tuple[InvalidCellRule | CoverageRegion, ...],
) -> bool:
    """Return whether one source is known to contain the whole schema cell."""
    return any(_predicate_set_implies(cell, source.predicates) for source in sources)


def _predicate_set_implies(
    premises: tuple[StructuredPredicate, ...],
    conclusions: tuple[StructuredPredicate, ...],
) -> bool:
    return all(
        _predicate_set_implies_predicate(premises, conclusion)
        for conclusion in conclusions
    )


def _predicate_set_implies_predicate(
    premises: tuple[StructuredPredicate, ...],
    conclusion: StructuredPredicate,
) -> bool:
    if isinstance(conclusion, MembershipPredicate):
        return any(
            isinstance(premise, MembershipPredicate)
            and premise.field == conclusion.field
            and premise.values <= conclusion.values
            for premise in premises
        )
    if isinstance(conclusion, EvidencePredicate):
        return any(
            isinstance(premise, EvidencePredicate)
            and premise.field == conclusion.field
            and premise.evidence <= conclusion.evidence
            for premise in premises
        )
    if isinstance(conclusion, ComparisonPredicate):
        comparisons = tuple(
            premise
            for premise in premises
            if isinstance(premise, ComparisonPredicate)
            and premise.field == conclusion.field
        )
        memberships = tuple(
            premise
            for premise in premises
            if isinstance(premise, MembershipPredicate)
            and premise.field == conclusion.field
        )
        return _comparison_predicates_imply(
            comparisons,
            memberships,
            conclusion,
        )
    return conclusion in premises


def _comparison_predicates_imply(
    comparisons: tuple[ComparisonPredicate, ...],
    memberships: tuple[MembershipPredicate, ...],
    conclusion: ComparisonPredicate,
) -> bool:
    if memberships:
        values = frozenset.intersection(
            *(membership.values for membership in memberships)
        )
        return bool(values) and all(
            conclusion.evaluate(
                ParameterDescriptor({conclusion.field: DescriptorCoordinate(value)})
            )
            for value in values
        )
    lower = max(
        (
            (
                comparison.value,
                comparison.operator in {">", "=="},
            )
            for comparison in comparisons
            if comparison.operator in {">", ">=", "=="}
        ),
        default=None,
    )
    upper = min(
        (
            (
                comparison.value,
                comparison.operator in {"<", "=="},
            )
            for comparison in comparisons
            if comparison.operator in {"<", "<=", "=="}
        ),
        default=None,
    )
    if conclusion.operator == ">":
        return lower is not None and (
            lower[0] > conclusion.value or (lower[0] == conclusion.value and lower[1])
        )
    if conclusion.operator == ">=":
        return lower is not None and lower[0] >= conclusion.value
    if conclusion.operator == "<":
        return upper is not None and (
            upper[0] < conclusion.value or (upper[0] == conclusion.value and upper[1])
        )
    if conclusion.operator == "<=":
        return upper is not None and upper[0] <= conclusion.value
    if conclusion.operator == "==":
        return (
            lower is not None
            and upper is not None
            and lower[0] == conclusion.value
            and upper[0] == conclusion.value
        )
    raise ValueError(f"unsupported comparison operator {conclusion.operator!r}")


def _schema_atlas_regions(
    schema: ParameterSpaceSchema,
    regions: tuple[CoverageRegion, ...] = (),
) -> tuple[_AtlasRegionSource, ...]:
    """Return every schema region that should appear in atlas projections."""
    return (
        *schema.invalid_cells,
        *regions,
        *_capability_atlas_uncovered_cells(schema, regions),
    )


def _capability_atlas_uncovered_cells(
    schema: ParameterSpaceSchema,
    regions: tuple[CoverageRegion, ...],
) -> tuple[_AtlasUncoveredCell, ...]:
    key = _atlas_schema_region_key(schema, regions)
    if key not in _ATLAS_UNCOVERED_CELLS_BY_SCHEMA:
        _ATLAS_UNCOVERED_CELLS_BY_SCHEMA[key] = (
            _compute_capability_atlas_uncovered_cells(
                schema,
                regions,
            )
        )
    return _ATLAS_UNCOVERED_CELLS_BY_SCHEMA[key]


def _compute_capability_atlas_uncovered_cells(
    schema: ParameterSpaceSchema,
    regions: tuple[CoverageRegion, ...],
) -> tuple[_AtlasUncoveredCell, ...]:
    cells = dict(_schema_indexed_cells(schema))
    return tuple(
        _AtlasUncoveredCell(cells[coordinates])
        for coordinates, signature in _schema_signature_cells(schema, regions)
        if _atlas_signature_has_source(signature, _AtlasUncoveredCell)
    )


def _schema_cells(
    schema: ParameterSpaceSchema,
) -> tuple[tuple[StructuredPredicate, ...], ...]:
    return tuple(cell for _coordinates, cell in _schema_indexed_cells(schema))


def _schema_indexed_cells(
    schema: ParameterSpaceSchema,
) -> tuple[tuple[tuple[int, ...], tuple[StructuredPredicate, ...]], ...]:
    key = tuple((axis.field, axis.bins) for axis in schema.axes)
    if key not in _SCHEMA_INDEXED_CELLS_BY_AXES:
        _SCHEMA_INDEXED_CELLS_BY_AXES[key] = _compute_schema_indexed_cells(schema)
    return _SCHEMA_INDEXED_CELLS_BY_AXES[key]


def _compute_schema_indexed_cells(
    schema: ParameterSpaceSchema,
) -> tuple[tuple[tuple[int, ...], tuple[StructuredPredicate, ...]], ...]:
    axis_cells = tuple(_axis_cells(axis) for axis in schema.axes)
    return tuple(
        (
            coordinates,
            tuple(
                predicate
                for axis_index, cell_index in enumerate(coordinates)
                for predicate in axis_cells[axis_index][cell_index]
            ),
        )
        for coordinates in product(
            *(tuple(range(len(axis_cell))) for axis_cell in axis_cells)
        )
    )


def _axis_cells(axis: ParameterAxis) -> tuple[tuple[StructuredPredicate, ...], ...]:
    key = (axis.field, axis.bins)
    if key not in _AXIS_CELLS_BY_AXIS:
        _AXIS_CELLS_BY_AXIS[key] = tuple(
            _axis_region_predicates(axis, axis_region) for axis_region in axis.bins
        )
    return _AXIS_CELLS_BY_AXIS[key]


def _axis_region_predicates(
    axis: ParameterAxis,
    axis_region: ParameterBin | NumericInterval,
) -> tuple[StructuredPredicate, ...]:
    if isinstance(axis_region, ParameterBin):
        return (MembershipPredicate(axis.field, axis_region.values),)
    predicates: list[StructuredPredicate] = []
    if axis_region.lower is not None:
        predicates.append(
            ComparisonPredicate(
                axis.field,
                ">=" if axis_region.include_lower else ">",
                axis_region.lower,
            )
        )
    if axis_region.upper is not None:
        predicates.append(
            ComparisonPredicate(
                axis.field,
                "<=" if axis_region.include_upper else "<",
                axis_region.upper,
            )
        )
    return tuple(predicates)


def _atlas_source_key(source: _AtlasRegionSource) -> tuple[object, ...]:
    if isinstance(source, _AtlasUncoveredCell):
        return (
            _AtlasUncoveredCell,
            tuple(_predicate_key(predicate) for predicate in source.predicates),
        )
    return (type(source), id(source))


def _atlas_cell_source(source: InvalidCellRule | CoverageRegion) -> _AtlasCellSource:
    if isinstance(source, CoverageRegion):
        return _AtlasCellSource(
            CoverageRegion,
            (
                source.owner,
                tuple(_predicate_key(predicate) for predicate in source.predicates),
            ),
        )
    return _AtlasCellSource(
        InvalidCellRule,
        (tuple(_predicate_key(predicate) for predicate in source.predicates),),
    )


def _atlas_uncovered_cell_source(
    coordinates: tuple[int, ...],
) -> _AtlasCellSource:
    return _AtlasCellSource(
        _AtlasUncoveredCell,
        coordinates,
    )


def _predicate_key(predicate: StructuredPredicate) -> tuple[object, ...]:
    if isinstance(predicate, ComparisonPredicate):
        return (
            ComparisonPredicate,
            predicate.field,
            predicate.operator,
            predicate.value,
            predicate.accepted_evidence,
        )
    if isinstance(predicate, MembershipPredicate):
        return (
            MembershipPredicate,
            predicate.field,
            predicate.values,
            predicate.accepted_evidence,
        )
    if isinstance(predicate, EvidencePredicate):
        return (EvidencePredicate, predicate.field, predicate.evidence)
    assert isinstance(predicate, AffineComparisonPredicate)
    return (
        AffineComparisonPredicate,
        tuple(sorted(predicate.terms.items(), key=lambda item: _field_label(item[0]))),
        predicate.operator,
        predicate.value,
        predicate.offset,
        predicate.accepted_evidence,
    )


def _predicate_sets_are_disjoint(
    left: tuple[StructuredPredicate, ...],
    right: tuple[StructuredPredicate, ...],
) -> bool:
    key = (id(left), id(right))
    reverse_key = (key[1], key[0])
    if reverse_key in _PREDICATE_SET_DISJOINTNESS:
        return _PREDICATE_SET_DISJOINTNESS[reverse_key]
    if key not in _PREDICATE_SET_DISJOINTNESS:
        _PREDICATE_SET_DISJOINTNESS[key] = predicate_sets_are_disjoint(left, right)
    return _PREDICATE_SET_DISJOINTNESS[key]


def _cell_is_provably_disjoint_from_predicate(
    cell: tuple[StructuredPredicate, ...],
    predicate: StructuredPredicate,
) -> bool:
    key = (tuple(id(cell_predicate) for cell_predicate in cell), id(predicate))
    if key not in _CELL_PREDICATE_DISJOINTNESS:
        _CELL_PREDICATE_DISJOINTNESS[key] = any(
            _predicates_are_disjoint(cell_predicate, predicate)
            for cell_predicate in cell
            if cell_predicate.referenced_fields & predicate.referenced_fields
        )
    return _CELL_PREDICATE_DISJOINTNESS[key]


def _predicates_are_disjoint(
    left: StructuredPredicate,
    right: StructuredPredicate,
) -> bool:
    if isinstance(left, MembershipPredicate) and isinstance(right, MembershipPredicate):
        return not left.values & right.values
    if isinstance(left, EvidencePredicate) and isinstance(right, EvidencePredicate):
        return not left.evidence & right.evidence
    if isinstance(left, EvidencePredicate) and isinstance(right, ComparisonPredicate):
        return not left.evidence & right.accepted_evidence
    if isinstance(left, ComparisonPredicate) and isinstance(right, EvidencePredicate):
        return not left.accepted_evidence & right.evidence
    if isinstance(left, MembershipPredicate) and isinstance(right, ComparisonPredicate):
        return not any(_predicate_accepts_value(right, value) for value in left.values)
    if isinstance(left, ComparisonPredicate) and isinstance(right, MembershipPredicate):
        return not any(_predicate_accepts_value(left, value) for value in right.values)
    if isinstance(left, ComparisonPredicate) and isinstance(right, ComparisonPredicate):
        return predicate_sets_are_disjoint((left,), (right,))
    return predicate_sets_are_disjoint((left,), (right,))


def _predicate_accepts_value(
    predicate: ComparisonPredicate,
    value: Any,
) -> bool:
    return predicate.evaluate(
        ParameterDescriptor({predicate.field: DescriptorCoordinate(value)})
    )


def _cell_is_provably_disjoint_from_source(
    cell: tuple[StructuredPredicate, ...],
    source: tuple[StructuredPredicate, ...],
) -> bool:
    simple_predicates = tuple(
        predicate
        for predicate in source
        if not isinstance(predicate, AffineComparisonPredicate)
    )
    return any(
        _cell_is_provably_disjoint_from_predicate(cell, predicate)
        for predicate in simple_predicates
    )


def _atlas_regions_for_schema(
    schema: ParameterSpaceSchema,
) -> tuple[CoverageRegion, ...]:
    return tuple(
        region
        for region in _capability_atlas_coverage_regions()
        if _region_inhabits_schema(region, schema)
    )


def _region_inhabits_schema(
    region: CoverageRegion,
    schema: ParameterSpaceSchema,
) -> bool:
    try:
        schema.validate_coverage_region(region)
    except ValueError:
        return False
    if schema.auxiliary_fields and region.referenced_fields <= schema.auxiliary_fields:
        return False
    return True


def _region_is_value_partitioned(region: CoverageRegion) -> bool:
    return not any(
        isinstance(predicate, EvidencePredicate) for predicate in region.predicates
    )


def _coverage_region_name(region: CoverageRegion) -> str:
    return region.owner.__name__


def _coverage_region_predicate_key(region: CoverageRegion) -> tuple[object, ...]:
    return tuple(_predicate_key(predicate) for predicate in region.predicates)


def _project_alternative_geometry(
    predicates: tuple[Any, ...],
    *,
    x_axis: ParameterAxis,
    y_axis: ParameterAxis,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    polygons: list[tuple[tuple[float, float], ...]] = []
    for x_index, x_cell in enumerate(_axis_cells(x_axis)):
        for y_index, y_cell in enumerate(_axis_cells(y_axis)):
            cell = x_cell + y_cell
            if any(
                _cell_is_provably_disjoint_from_predicate(cell, predicate)
                for predicate in predicates
            ):
                continue
            polygons.append(
                (
                    (float(x_index), float(y_index)),
                    (float(x_index + 1), float(y_index)),
                    (float(x_index + 1), float(y_index + 1)),
                    (float(x_index), float(y_index + 1)),
                )
            )
    return tuple(polygons)


_AtlasProjectedRegion: TypeAlias = tuple[
    _AtlasRegionSource,
    tuple[Any, ...],
    tuple[tuple[float, float], ...],
]


def _projected_region_shapes(
    group: _AtlasDescriptorGroup,
) -> tuple[_AtlasProjectedRegion, ...]:
    schema = _atlas_group_schema(group)
    regions = _atlas_regions_for_schema(schema)
    x_axis = _atlas_schema_axis(schema, _atlas_group_x_axis(group))
    y_axis = _atlas_schema_axis(schema, _atlas_group_y_axis(group))
    shapes: list[_AtlasProjectedRegion] = []
    for source in _schema_atlas_regions(schema, regions):
        alternatives = _source_alternatives(schema, source, regions)
        for predicates in alternatives:
            geometry = _project_alternative_geometry(
                predicates,
                x_axis=x_axis,
                y_axis=y_axis,
            )
            if not geometry:
                axes = (_atlas_axis_field(x_axis), _atlas_axis_field(y_axis))
                raise AssertionError(
                    f"atlas region {_atlas_source_label(source)!r} has no "
                    f"visible projection onto {axes!r}"
                )
            for points in geometry:
                shapes.append((source, predicates, points))
    return tuple(shapes)


def _atlas_source_label(
    source: _AtlasRegionSource,
) -> _AtlasText:
    if isinstance(source, CoverageRegion):
        return _AtlasText(_coverage_region_name(source))
    if isinstance(source, _AtlasUncoveredCell):
        return _AtlasText("schema cell")
    return _AtlasText(source.name)


def _atlas_source_status(
    source: _AtlasRegionSource,
) -> _AtlasText:
    if isinstance(source, InvalidCellRule):
        return _AtlasText("invalid")
    if isinstance(source, CoverageRegion):
        return _AtlasText("owned")
    return _AtlasText("uncovered")


class _CapabilityAtlasDocClaim(Claim[None]):
    """Claim: capability atlas documentation can be generated."""

    @property
    def description(self) -> str:
        return "algorithm_capabilities/capability_atlas_doc_generates"

    def check(self, _calibration: None) -> None:
        self._assert_no_atlas_dataclass_stores_presentation_text()
        self._assert_uncovered_regions_are_computed()
        self._assert_axis_views_are_schema_partitions()
        self._assert_cell_signatures_preserve_source_identity()
        self._assert_projection_axes_minimize_information_loss()
        self._assert_projection_axis_roles_are_derived()
        self._assert_evidence_schema_is_derived()
        self._assert_evidence_is_descriptors()
        self._assert_coverage_regions_are_schema_discovered()
        self._assert_descriptor_groups_are_schema_equivalence_classes()
        self._assert_docs_render_schema_hierarchy()
        self._assert_time_integrator_quantitative_evidence_is_plotted()
        self._assert_dense_linear_evidence_is_owned()
        self._assert_primitive_solve_projection_gaps_are_intentional()
        self._assert_fully_constrained_reaction_network_is_owned()
        self._assert_nonlinear_root_without_jacobian_is_visible_gap()
        self._assert_directional_derivative_root_is_owned()
        for group in _capability_atlas_descriptor_groups():
            schema = _atlas_group_schema(group)
            regions = _atlas_regions_for_schema(schema)
            discovered = _schema_atlas_regions(schema, regions)
            uncovered = _capability_atlas_uncovered_cells(schema, regions)
            assert {_atlas_source_key(source) for source in discovered} == {
                _atlas_source_key(source)
                for source in (*schema.invalid_cells, *regions, *uncovered)
            }
            shapes = _projected_region_shapes(group)
            assert shapes
            for source, _predicates, points in shapes:
                assert _atlas_source_label(source)
                assert points

    @staticmethod
    def _assert_docs_render_schema_hierarchy() -> None:
        from scripts.gen_capability_atlas_docs import render_capability_atlas

        rendered = render_capability_atlas()
        assert "## Parameter Space Hierarchy" in rendered
        assert "## Projection Plots" in rendered
        assert "`eigenpair_residual`: valid spectral solve-relation evidence" not in (
            rendered
        )
        assert rendered.index("## Parameter Space Hierarchy") < rendered.index(
            "## Projection Plots"
        )
        assert rendered.index("## Computed Gaps") < rendered.index(
            "## Descriptor Evidence Overlay"
        )
        for schema in _capability_atlas_schemas():
            assert f"### {schema.name}" in rendered
            assert "- Primitive axes:" in rendered
            for axis in schema.axes:
                assert f"`{_field_label(axis.field)}`" in rendered
            for field in schema.auxiliary_fields:
                assert f"`{_field_label(field)}`" in rendered
            for region in schema.derived_regions:
                assert f"`{region.name}`" in rendered
        assert "## Descriptor Evidence Overlay" in rendered

    @staticmethod
    def _assert_time_integrator_quantitative_evidence_is_plotted() -> None:
        schema = map_structure_parameter_schema()
        regions = _atlas_regions_for_schema(schema)
        derived = {region.name: region for region in schema.derived_regions}
        nonstiff, stiff, domain_limited = _atlas_step_diagnostic_descriptors()
        for descriptor in (nonstiff, stiff, domain_limited):
            schema.validate_descriptor(descriptor)
            assert schema.cell_status(descriptor, regions) == "owned"
            assert any(
                descriptor in group for group in _capability_atlas_descriptor_groups()
            )
        assert derived["nonstiff_step"].contains(nonstiff)
        assert derived["stiff_step"].contains(stiff)
        assert derived["domain_limited_step"].contains(domain_limited)
        assert all(
            descriptor.coordinate(MapStructureField.RHS_DERIVATIVE_ORACLE_KIND).value
            == "jacobian_callback"
            for descriptor in (nonstiff, stiff, domain_limited)
        )
        assert any(
            descriptor.coordinate(MapStructureField.RHS_EVALUATION_COST_FMAS).value
            > 0.0
            and descriptor.coordinate(
                MapStructureField.NORDSIECK_HISTORY_AVAILABLE
            ).value
            is False
            for descriptor in _capability_atlas_descriptors()
            if _atlas_schema_for_descriptor(descriptor) == schema
        )

    @staticmethod
    def _assert_dense_linear_evidence_is_owned() -> None:
        schema = linear_solver_parameter_schema()
        regions = _atlas_regions_for_schema(schema)
        dense_descriptors = tuple(
            descriptor
            for descriptor in _capability_atlas_descriptors()
            if _atlas_schema_for_descriptor(descriptor) == schema
            and descriptor.coordinate(
                LinearSolverField.LINEAR_OPERATOR_MATRIX_AVAILABLE
            ).value
        )
        assert dense_descriptors
        assert all(
            descriptor.coordinate(
                DecompositionField.FACTORIZATION_WORK_BUDGET_FMAS
            ).evidence
            != "unavailable"
            for descriptor in dense_descriptors
            if schema.cell_status(descriptor, regions) != "invalid"
        )
        assert all(
            schema.cell_status(descriptor, regions) == "owned"
            for descriptor in dense_descriptors
            if schema.cell_status(descriptor, regions) != "invalid"
        )

    @staticmethod
    def _assert_primitive_solve_projection_gaps_are_intentional() -> None:
        schema = solve_relation_parameter_schema()
        regions = _atlas_regions_for_schema(schema)
        descriptors = tuple(
            descriptor
            for descriptor in _capability_atlas_descriptors()
            if _atlas_schema_for_descriptor(descriptor) == schema
        )
        assert any(
            descriptor.coordinate(SolveRelationField.ACCEPTANCE_RELATION).value
            == "eigenpair_residual"
            and schema.cell_status(descriptor, regions) == "owned"
            for descriptor in descriptors
        )
        assert not any(
            descriptor.coordinate(SolveRelationField.ACCEPTANCE_RELATION).value
            == "eigenpair_residual"
            and schema.cell_status(descriptor, regions) == "uncovered"
            for descriptor in descriptors
        )
        assert not any(
            descriptor.coordinate(SolveRelationField.OBJECTIVE_RELATION).value
            == "least_squares"
            and schema.cell_status(descriptor, regions) == "uncovered"
            for descriptor in descriptors
        )

    @staticmethod
    def _assert_fully_constrained_reaction_network_is_owned() -> None:
        schema = reaction_network_parameter_schema()
        regions = _atlas_regions_for_schema(schema)
        descriptor = _SolveRelationSchemaClaim._reaction_network_descriptor()
        schema.validate_descriptor(descriptor)
        assert schema.cell_status(descriptor, regions) == "owned"
        assert schema.covering_region(descriptor, regions).owner is (
            NuclearStatisticalEquilibriumSolver
        )

    @staticmethod
    def _assert_nonlinear_root_without_jacobian_is_visible_gap() -> None:
        schema = solve_relation_parameter_schema()
        regions = _atlas_regions_for_schema(schema)
        descriptor = _SolveRelationSchemaClaim._solve_descriptor(
            target_is_zero=True,
            map_linearity_defect=1.0,
            matrix_representation_available=False,
            derivative_oracle_kind="none",
        )
        schema.validate_descriptor(descriptor)
        assert schema.cell_status(descriptor, regions) == "uncovered"
        assert any(
            uncovered.contains(descriptor)
            for uncovered in _capability_atlas_uncovered_cells(schema, regions)
        )

    @staticmethod
    def _assert_directional_derivative_root_is_owned() -> None:
        schema = solve_relation_parameter_schema()
        regions = _atlas_regions_for_schema(schema)
        descriptor = _SolveRelationSchemaClaim._solve_descriptor(
            target_is_zero=True,
            map_linearity_defect=1.0,
            matrix_representation_available=False,
            derivative_oracle_kind="jvp",
        )
        schema.validate_descriptor(descriptor)
        assert schema.cell_status(descriptor, regions) == "owned"

    @staticmethod
    def _assert_no_atlas_dataclass_stores_presentation_text() -> None:
        tree = ast.parse(Path(__file__).read_text())
        presentation_text_aliases = _CapabilityAtlasDocClaim._presentation_text_aliases(
            tree
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not _CapabilityAtlasDocClaim._is_dataclass(node):
                continue
            annotations = (
                child.annotation
                for child in node.body
                if isinstance(child, ast.AnnAssign)
            )
            assert not any(
                _CapabilityAtlasDocClaim._annotation_mentions_presentation_text(
                    annotation,
                    presentation_text_aliases,
                )
                for annotation in annotations
            )

    @staticmethod
    def _presentation_text_aliases(tree: ast.Module) -> frozenset[str]:
        aliases: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            if not (
                isinstance(node.value.func, ast.Name)
                and node.value.func.id == "NewType"
            ):
                continue
            if len(node.value.args) != 2:
                continue
            if not (
                isinstance(node.value.args[1], ast.Name)
                and node.value.args[1].id == "str"
            ):
                continue
            aliases.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        return frozenset(aliases)

    @staticmethod
    def _is_dataclass(node: ast.ClassDef) -> bool:
        return any(
            (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "dataclass"
            )
            or (isinstance(decorator, ast.Name) and decorator.id == "dataclass")
            for decorator in node.decorator_list
        )

    @staticmethod
    def _annotation_mentions_presentation_text(
        annotation: ast.AST,
        presentation_text_aliases: frozenset[str],
    ) -> bool:
        return any(
            isinstance(node, ast.Name) and node.id in presentation_text_aliases
            for node in ast.walk(annotation)
        )

    @staticmethod
    def _assert_uncovered_regions_are_computed() -> None:
        for schema in _capability_atlas_schemas():
            regions = _atlas_regions_for_schema(schema)
            uncovered = _capability_atlas_uncovered_cells(schema, regions)
            indexed_cells = dict(_schema_indexed_cells(schema))
            signature_uncovered = tuple(
                _AtlasUncoveredCell(indexed_cells[coordinates])
                for coordinates, signature in _schema_signature_cells(schema, regions)
                if _atlas_signature_has_source(signature, _AtlasUncoveredCell)
            )
            assert uncovered == signature_uncovered
            for source in uncovered:
                assert not _cell_has_covering_source(
                    source.predicates, (*schema.invalid_cells, *regions)
                )

    @staticmethod
    def _assert_axis_views_are_schema_partitions() -> None:
        for schema in _capability_atlas_schemas():
            for axis in schema.axes:
                view = _atlas_axis_view(axis)
                assert view.cells == axis.bins
                assert view.use_log_scale == _axis_uses_log_scale(axis)
                assert _atlas_axis_range(axis) == (0.0, float(len(axis.bins)))
        for group in _capability_atlas_descriptor_groups():
            schema = _atlas_group_schema(group)
            x_axis = _atlas_schema_axis(schema, _atlas_group_x_axis(group))
            y_axis = _atlas_schema_axis(schema, _atlas_group_y_axis(group))
            assert _atlas_group_x_range(group) == _atlas_axis_range(x_axis)
            assert _atlas_group_y_range(group) == _atlas_axis_range(y_axis)

    @staticmethod
    def _assert_cell_signatures_preserve_source_identity() -> None:
        for schema in _capability_atlas_schemas():
            regions = _atlas_regions_for_schema(schema)
            signatures = tuple(
                signature
                for _coordinates, signature in _schema_signature_cells(schema, regions)
            )
            sources = frozenset(
                source for signature in signatures for source in signature
            )
            assert all(source.kind is not type for source in sources)
            assert {source for source in sources if source.kind is CoverageRegion} == {
                _atlas_cell_source(region)
                for region in regions
                if _region_is_value_partitioned(region)
            }
            assert {
                source for source in sources if source.kind is InvalidCellRule
            }.issubset({_atlas_cell_source(rule) for rule in schema.invalid_cells})
            assert all(
                isinstance(source.key, tuple) and source.key
                for source in sources
                if source.kind is _AtlasUncoveredCell
            )

    @staticmethod
    def _assert_projection_axes_minimize_information_loss() -> None:
        for group in _capability_atlas_descriptor_groups():
            schema = _atlas_group_schema(group)
            regions = _atlas_regions_for_schema(schema)
            projection = _atlas_group_projection(group)
            candidates = _atlas_candidate_projections(schema, regions)
            assert projection in candidates
            assert projection.objective == min(
                candidate.objective for candidate in candidates
            )
            assert {_atlas_group_x_axis(group), _atlas_group_y_axis(group)} == {
                _atlas_axis_field(projection.x_axis),
                _atlas_axis_field(projection.y_axis),
            }

    @classmethod
    def _assert_descriptor_groups_are_schema_equivalence_classes(cls) -> None:
        groups = _capability_atlas_descriptor_groups()
        plotted = tuple(descriptor for group in groups for descriptor in group)
        assert plotted == _capability_atlas_descriptors()
        for group in groups:
            assert group
            schema = _atlas_schema_for_descriptor(group[0])
            assert all(
                _atlas_schema_for_descriptor(descriptor) == schema
                for descriptor in group
            )
        for left in plotted:
            for right in plotted:
                assert (
                    _atlas_schema_for_descriptor(left)
                    == _atlas_schema_for_descriptor(right)
                ) == (any(left in group and right in group for group in groups))

    @classmethod
    def _assert_evidence_schema_is_derived(cls) -> None:
        for descriptor in _capability_atlas_descriptors():
            _atlas_schema_for_descriptor(descriptor)

    @classmethod
    def _assert_evidence_is_descriptors(cls) -> None:
        assert all(
            isinstance(descriptor, ParameterDescriptor)
            for descriptor in _capability_atlas_descriptors()
        )

    @classmethod
    def _assert_coverage_regions_are_schema_discovered(cls) -> None:
        for schema in _capability_atlas_schemas():
            assert _atlas_regions_for_schema(schema) == tuple(
                region
                for region in _capability_atlas_coverage_regions()
                if _region_inhabits_schema(region, schema)
            )

    @classmethod
    def _assert_projection_axis_roles_are_derived(cls) -> None:
        for group in _capability_atlas_descriptor_groups():
            shown = {_atlas_group_x_axis(group), _atlas_group_y_axis(group)}
            fixed = set(_atlas_fixed_axes(group))
            marginalized = set(_atlas_marginalized_axes(group))
            assert shown | fixed | marginalized == {
                _atlas_axis_field(axis) for axis in _atlas_group_schema(group).axes
            }
            assert not (shown & fixed)
            assert not (shown & marginalized)
            assert not (fixed & marginalized)
