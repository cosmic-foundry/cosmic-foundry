"""Shared algorithm capability declarations and selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, TypeAlias

import numpy as np

from cosmic_foundry.computation.tensor import Tensor


@dataclass(frozen=True)
class AlgorithmStructureContract:
    """Required input structure and provided algorithmic properties."""

    requires: frozenset[str]
    provides: frozenset[str]


@dataclass(frozen=True)
class AlgorithmCapability:
    """Declared capability of one selectable algorithm implementation."""

    name: str
    implementation: str
    category: str
    contract: AlgorithmStructureContract
    min_order: int | None = None
    max_order: int | None = None
    order_step: int | None = None
    supported_orders: frozenset[int] | None = None
    coverage_regions: tuple[CoverageRegion, ...] = ()

    def supports(self, request: AlgorithmRequest) -> bool:
        """Return whether this declaration inhabits ``request``."""
        if request.order is not None:
            if self.supported_orders is not None:
                if request.order not in self.supported_orders:
                    return False
            else:
                if self.min_order is not None and request.order < self.min_order:
                    return False
                if self.max_order is not None and request.order > self.max_order:
                    return False
                if self.order_step is not None:
                    if self.min_order is None:
                        return False
                    if (request.order - self.min_order) % self.order_step != 0:
                        return False
        if not request.requested_properties <= self.contract.provides:
            return False
        structure_satisfied = self.contract.requires <= request.available_structure
        descriptor_satisfied = not self.coverage_regions or (
            request.descriptor is not None
            and any(
                region.contains(request.descriptor) for region in self.coverage_regions
            )
        )
        return structure_satisfied and descriptor_satisfied


@dataclass(frozen=True)
class AlgorithmRequest:
    """Requested input structure and desired algorithmic properties."""

    available_structure: frozenset[str] = frozenset()
    requested_properties: frozenset[str] = frozenset()
    order: int | None = None
    descriptor: ParameterDescriptor | None = None


class AlgorithmRegistry:
    """Select algorithm implementations by declared capabilities."""

    def __init__(self, capabilities: tuple[AlgorithmCapability, ...]) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> tuple[AlgorithmCapability, ...]:
        """Registered implementation declarations."""
        return self._capabilities

    def matching(self, request: AlgorithmRequest) -> tuple[AlgorithmCapability, ...]:
        """Return all declarations that inhabit ``request``."""
        return tuple(cap for cap in self._capabilities if cap.supports(request))

    def select(self, request: AlgorithmRequest) -> AlgorithmCapability:
        """Return the unique implementation inhabiting ``request``."""
        matches = self.matching(request)
        if not matches:
            raise ValueError(f"no algorithm satisfies request {request!r}")
        if len(matches) == 1:
            return matches[0]

        names = ", ".join(cap.name for cap in matches)
        raise ValueError(f"ambiguous algorithm request {request!r}: {names}")


ScalarValue: TypeAlias = bool | int | float | str
EvidenceSource: TypeAlias = Literal[
    "exact",
    "upper_bound",
    "lower_bound",
    "estimate",
    "caller_assumption",
    "unavailable",
]
ComparisonOperator: TypeAlias = Literal["<", "<=", "==", "!=", ">=", ">"]
CellStatus: TypeAlias = Literal["invalid", "owned", "uncovered"]


class SolveRelationField(Enum):
    """Schema-owned descriptor fields for primitive solve relations."""

    ACCEPTANCE_RELATION = "acceptance_relation"
    AUXILIARY_SCALAR_COUNT = "auxiliary_scalar_count"
    BACKEND_KIND = "backend_kind"
    DERIVATIVE_ORACLE_KIND = "derivative_oracle_kind"
    DEVICE_KIND = "device_kind"
    DIM_X = "dim_x"
    DIM_Y = "dim_y"
    EQUALITY_CONSTRAINT_COUNT = "equality_constraint_count"
    MAP_LINEARITY_DEFECT = "map_linearity_defect"
    MATRIX_REPRESENTATION_AVAILABLE = "matrix_representation_available"
    MEMORY_BUDGET_BYTES = "memory_budget_bytes"
    NORMALIZATION_CONSTRAINT_COUNT = "normalization_constraint_count"
    OBJECTIVE_RELATION = "objective_relation"
    OPERATOR_APPLICATION_AVAILABLE = "operator_application_available"
    REQUESTED_RESIDUAL_TOLERANCE = "requested_residual_tolerance"
    REQUESTED_SOLUTION_TOLERANCE = "requested_solution_tolerance"
    RESIDUAL_TARGET_AVAILABLE = "residual_target_available"
    TARGET_IS_ZERO = "target_is_zero"
    WORK_BUDGET_FMAS = "work_budget_fmas"


class LinearSolverField(Enum):
    """Schema-owned descriptor fields for linear-operator solve coverage."""

    ASSEMBLY_COST_FMAS = "assembly_cost_fmas"
    COERCIVITY_LOWER_BOUND = "coercivity_lower_bound"
    CONDITION_ESTIMATE = "condition_estimate"
    DIAGONAL_DOMINANCE_MARGIN = "diagonal_dominance_margin"
    DIAGONAL_NONZERO_MARGIN = "diagonal_nonzero_margin"
    LINEAR_OPERATOR_MATRIX_AVAILABLE = "linear_operator_matrix_available"
    LINEAR_OPERATOR_MEMORY_BYTES = "linear_operator_memory_bytes"
    MATVEC_COST_FMAS = "matvec_cost_fmas"
    NULLITY_ESTIMATE = "nullity_estimate"
    RANK_ESTIMATE = "rank_estimate"
    RHS_CONSISTENCY_DEFECT = "rhs_consistency_defect"
    SINGULAR_VALUE_LOWER_BOUND = "singular_value_lower_bound"
    SKEW_SYMMETRY_DEFECT = "skew_symmetry_defect"
    SYMMETRY_DEFECT = "symmetry_defect"


class DecompositionField(Enum):
    """Schema-owned descriptor fields for decomposition coverage."""

    CONDITION_ESTIMATE = "decomposition_condition_estimate"
    FACTORIZATION_MEMORY_BUDGET_BYTES = "factorization_memory_budget_bytes"
    FACTORIZATION_WORK_BUDGET_FMAS = "factorization_work_budget_fmas"
    MATRIX_COLUMNS = "matrix_columns"
    MATRIX_NULLITY_ESTIMATE = "matrix_nullity_estimate"
    MATRIX_RANK_ESTIMATE = "matrix_rank_estimate"
    MATRIX_ROWS = "matrix_rows"
    SINGULAR_VALUE_LOWER_BOUND = "decomposition_singular_value_lower_bound"


class ReactionNetworkField(Enum):
    """Schema-owned descriptor fields for stoichiometric reaction networks."""

    CONSERVATION_LAW_COUNT = "conservation_law_count"
    EQUILIBRIUM_CONSTRAINT_COUNT = "equilibrium_constraint_count"
    REACTION_COUNT = "reaction_count"
    SPECIES_COUNT = "species_count"
    STOICHIOMETRY_RANK = "stoichiometry_rank"


class MapStructureField(Enum):
    """Schema-owned descriptor fields for ODE map/operator structure."""

    ADDITIVE_COMPONENT_COUNT = "additive_component_count"
    EXACT_LINEAR_OPERATOR_AVAILABLE = "exact_linear_operator_available"
    EXPLICIT_COMPONENT_AVAILABLE = "explicit_component_available"
    HAMILTONIAN_PARTITION_AVAILABLE = "hamiltonian_partition_available"
    IMPLICIT_COMPONENT_AVAILABLE = "implicit_component_available"
    IMPLICIT_COMPONENT_DERIVATIVE_ORACLE_KIND = (
        "implicit_component_derivative_oracle_kind"
    )
    NORDSIECK_HISTORY_AVAILABLE = "nordsieck_history_available"
    NONLINEAR_RESIDUAL_AVAILABLE = "nonlinear_residual_available"
    RHS_HISTORY_AVAILABLE = "rhs_history_available"
    RHS_EVALUATION_AVAILABLE = "rhs_evaluation_available"
    SYMPLECTIC_FORM_INVARIANT_AVAILABLE = "symplectic_form_invariant_available"


DescriptorField: TypeAlias = (
    str
    | SolveRelationField
    | LinearSolverField
    | DecompositionField
    | ReactionNetworkField
    | MapStructureField
)


def _field_label(field: DescriptorField) -> str:
    if isinstance(
        field,
        SolveRelationField
        | LinearSolverField
        | DecompositionField
        | ReactionNetworkField
        | MapStructureField,
    ):
        return str(field.value)
    return field


@dataclass(frozen=True)
class ParameterBin:
    """Finite categorical bin for one parameter-space axis."""

    label: str
    values: frozenset[ScalarValue]

    def contains(self, value: ScalarValue) -> bool:
        """Return whether ``value`` inhabits this bin."""
        return value in self.values


@dataclass(frozen=True)
class NumericInterval:
    """Closed or half-open numeric interval for one parameter-space axis."""

    label: str
    lower: float | None = None
    upper: float | None = None
    include_lower: bool = True
    include_upper: bool = True

    def contains(self, value: ScalarValue) -> bool:
        """Return whether ``value`` inhabits this interval."""
        if isinstance(value, bool | str):
            return False
        numeric = float(value)
        if self.lower is not None:
            if self.include_lower:
                if numeric < self.lower:
                    return False
            elif numeric <= self.lower:
                return False
        if self.upper is not None:
            if self.include_upper:
                if numeric > self.upper:
                    return False
            elif numeric >= self.upper:
                return False
        return True


@dataclass(frozen=True)
class ParameterAxis:
    """Declared coordinate axis in a parameter-space schema."""

    field: DescriptorField
    bins: tuple[ParameterBin | NumericInterval, ...]
    units: str | None = None

    @property
    def label(self) -> str:
        """Human-readable field label for diagnostics and rendering."""
        return _field_label(self.field)

    def contains(self, coordinate: DescriptorCoordinate) -> bool:
        """Return whether ``coordinate`` belongs to one of the axis regions."""
        if coordinate.value is None:
            return False
        return any(
            bin_or_interval.contains(coordinate.value) for bin_or_interval in self.bins
        )


@dataclass(frozen=True)
class DescriptorCoordinate:
    """Value plus evidence state for one descriptor coordinate."""

    value: ScalarValue | None
    evidence: EvidenceSource = "exact"

    @property
    def known(self) -> bool:
        """Return whether the coordinate carries a concrete value."""
        return self.value is not None and self.evidence != "unavailable"


@dataclass(frozen=True)
class ParameterDescriptor:
    """Concrete problem location in descriptor coordinates."""

    coordinates: dict[DescriptorField, DescriptorCoordinate]

    def coordinate(self, field: DescriptorField) -> DescriptorCoordinate:
        """Return the coordinate for ``field`` or an explicit unavailable value."""
        if field in self.coordinates:
            return self.coordinates[field]
        if isinstance(field, str):
            for candidate, coordinate in self.coordinates.items():
                if _field_label(candidate) == field:
                    return coordinate
        return DescriptorCoordinate(None, evidence="unavailable")


class ParameterPredicate(Protocol):
    """Structured predicate over descriptor coordinates."""

    referenced_fields: frozenset[DescriptorField]

    def evaluate(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether ``descriptor`` satisfies the predicate."""
        ...


class SmallLinearOperator(Protocol):
    """Linear operator protocol needed for deterministic descriptor assembly."""

    def apply(self, u: Tensor) -> Tensor:
        """Apply the operator to one vector."""
        ...


@dataclass(frozen=True)
class LinearOperatorDescriptor:
    """Small assembled linear-operator descriptor plus matrix witness.

    The parameter descriptor is the schema-level object consumed by selection
    and atlas code.  The assembled matrix is retained as a deterministic witness
    for structural tests and documentation generators; it is not a performance
    timing result.
    """

    parameter_descriptor: ParameterDescriptor
    matrix: tuple[tuple[float, ...], ...]

    def coordinate(self, field: DescriptorField) -> DescriptorCoordinate:
        """Return the coordinate for ``field``."""
        return self.parameter_descriptor.coordinate(field)


def _compare(
    left: ScalarValue,
    operator: ComparisonOperator,
    right: ScalarValue,
) -> bool:
    if operator == "<":
        return left < right  # type: ignore[operator]
    if operator == "<=":
        return left <= right  # type: ignore[operator]
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == ">=":
        return left >= right  # type: ignore[operator]
    if operator == ">":
        return left > right  # type: ignore[operator]
    raise AssertionError(f"unsupported comparison operator {operator!r}")


@dataclass(frozen=True)
class ComparisonPredicate:
    """Comparison against one descriptor field."""

    field: DescriptorField
    operator: ComparisonOperator
    value: ScalarValue
    accepted_evidence: frozenset[EvidenceSource] = frozenset(
        {"exact", "upper_bound", "lower_bound", "estimate"}
    )

    @property
    def referenced_fields(self) -> frozenset[DescriptorField]:
        """Descriptor fields referenced by this predicate."""
        return frozenset({self.field})

    def evaluate(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether ``descriptor`` satisfies the comparison."""
        coordinate = descriptor.coordinate(self.field)
        if not coordinate.known or coordinate.evidence not in self.accepted_evidence:
            return False
        assert coordinate.value is not None
        return _compare(coordinate.value, self.operator, self.value)


@dataclass(frozen=True)
class MembershipPredicate:
    """Finite membership predicate over one descriptor field."""

    field: DescriptorField
    values: frozenset[ScalarValue]
    accepted_evidence: frozenset[EvidenceSource] = frozenset(
        {"exact", "caller_assumption"}
    )

    @property
    def referenced_fields(self) -> frozenset[DescriptorField]:
        """Descriptor fields referenced by this predicate."""
        return frozenset({self.field})

    def evaluate(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether ``descriptor`` has a member value."""
        coordinate = descriptor.coordinate(self.field)
        if not coordinate.known or coordinate.evidence not in self.accepted_evidence:
            return False
        return coordinate.value in self.values


@dataclass(frozen=True)
class AffineComparisonPredicate:
    """Affine comparison over numeric descriptor fields."""

    terms: Mapping[DescriptorField, float]
    operator: ComparisonOperator
    value: float
    offset: float = 0.0
    accepted_evidence: frozenset[EvidenceSource] = frozenset(
        {"exact", "upper_bound", "lower_bound", "estimate"}
    )

    @property
    def referenced_fields(self) -> frozenset[DescriptorField]:
        """Descriptor fields referenced by this predicate."""
        return frozenset(self.terms)

    def evaluate(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether the descriptor satisfies the affine bound."""
        total = self.offset
        for field, coefficient in self.terms.items():
            coordinate = descriptor.coordinate(field)
            if (
                not coordinate.known
                or coordinate.evidence not in self.accepted_evidence
                or isinstance(coordinate.value, bool | str)
            ):
                return False
            assert coordinate.value is not None
            total += coefficient * float(coordinate.value)
        return _compare(total, self.operator, self.value)


@dataclass(frozen=True)
class EvidencePredicate:
    """Evidence-state predicate over one descriptor field."""

    field: DescriptorField
    evidence: frozenset[EvidenceSource]

    @property
    def referenced_fields(self) -> frozenset[DescriptorField]:
        """Descriptor fields referenced by this predicate."""
        return frozenset({self.field})

    def evaluate(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether the coordinate carries one accepted evidence state."""
        return descriptor.coordinate(self.field).evidence in self.evidence


StructuredPredicate: TypeAlias = (
    ComparisonPredicate
    | MembershipPredicate
    | AffineComparisonPredicate
    | EvidencePredicate
)


@dataclass(frozen=True)
class DerivedParameterRegion:
    """Named region derived from primitive parameter-space coordinates."""

    name: str
    alternatives: tuple[tuple[StructuredPredicate, ...], ...]

    @property
    def referenced_fields(self) -> frozenset[DescriptorField]:
        """Descriptor fields referenced by this derived region."""
        fields: set[DescriptorField] = set()
        for alternative in self.alternatives:
            for predicate in alternative:
                fields.update(predicate.referenced_fields)
        return frozenset(fields)

    def contains(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether ``descriptor`` inhabits at least one alternative."""
        return any(
            all(predicate.evaluate(descriptor) for predicate in alternative)
            for alternative in self.alternatives
        )


@dataclass(frozen=True)
class InvalidCellRule:
    """Predicate-defined region that is outside the schema's valid domain."""

    name: str
    predicates: tuple[StructuredPredicate, ...]
    reason: str

    @property
    def referenced_fields(self) -> frozenset[DescriptorField]:
        """Descriptor fields referenced by this invalid-cell rule."""
        return frozenset().union(
            *(predicate.referenced_fields for predicate in self.predicates)
        )

    def matches(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether ``descriptor`` inhabits this invalid region."""
        return all(predicate.evaluate(descriptor) for predicate in self.predicates)


@dataclass(frozen=True)
class CoverageRegion:
    """Owned parameter-space region claimed by one implementation."""

    owner: type
    predicates: tuple[StructuredPredicate, ...]

    @property
    def referenced_fields(self) -> frozenset[DescriptorField]:
        """Descriptor fields referenced by this coverage region."""
        return frozenset().union(
            *(predicate.referenced_fields for predicate in self.predicates)
        )

    def contains(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether ``descriptor`` lies inside this region."""
        return all(predicate.evaluate(descriptor) for predicate in self.predicates)


def predicate_sets_are_disjoint(
    left: tuple[StructuredPredicate, ...],
    right: tuple[StructuredPredicate, ...],
) -> bool:
    """Return whether two predicate conjunctions are provably disjoint."""
    fields = frozenset().union(
        *(predicate.referenced_fields for predicate in left + right)
    )
    return any(
        _field_predicates_are_disjoint(
            field,
            tuple(
                predicate for predicate in left if field in predicate.referenced_fields
            ),
            tuple(
                predicate for predicate in right if field in predicate.referenced_fields
            ),
        )
        for field in fields
    ) or _affine_predicates_are_disjoint(left + right)


def coverage_regions_are_disjoint(regions: tuple[CoverageRegion, ...]) -> bool:
    """Return whether coverage regions form a pairwise-disjoint partition."""
    for pair_index, left in enumerate(regions):
        for right in regions[pair_index + 1 :]:
            if not predicate_sets_are_disjoint(left.predicates, right.predicates):
                return False
    return True


def _field_predicates_are_disjoint(
    field: DescriptorField,
    left: tuple[StructuredPredicate, ...],
    right: tuple[StructuredPredicate, ...],
) -> bool:
    predicates = left + right
    memberships = tuple(
        predicate
        for predicate in predicates
        if isinstance(predicate, MembershipPredicate)
    )
    evidences = tuple(
        predicate
        for predicate in predicates
        if isinstance(predicate, EvidencePredicate)
    )
    comparisons = tuple(
        predicate
        for predicate in predicates
        if isinstance(predicate, ComparisonPredicate)
    )
    if len(memberships) > 1:
        values = frozenset.intersection(
            *(predicate.values for predicate in memberships)
        )
        if not values:
            return True
    if len(evidences) > 1:
        evidence = frozenset.intersection(
            *(predicate.evidence for predicate in evidences)
        )
        if not evidence:
            return True
    if evidences and comparisons:
        evidence = frozenset.intersection(
            *(predicate.evidence for predicate in evidences),
            *(predicate.accepted_evidence for predicate in comparisons),
        )
        if not evidence:
            return True
    if memberships and comparisons:
        values = frozenset.intersection(
            *(predicate.values for predicate in memberships)
        )
        if not any(
            _value_satisfies_comparisons(field, value, comparisons) for value in values
        ):
            return True
    return _comparisons_are_disjoint(field, comparisons)


def _comparisons_are_disjoint(
    field: DescriptorField,
    comparisons: tuple[ComparisonPredicate, ...],
) -> bool:
    equal_values = {
        predicate.value for predicate in comparisons if predicate.operator == "=="
    }
    if len(equal_values) > 1:
        return True
    if equal_values:
        value = next(iter(equal_values))
        return not _value_satisfies_comparisons(field, value, comparisons)
    lower_bound = _strongest_lower_bound(comparisons)
    upper_bound = _strongest_upper_bound(comparisons)
    if lower_bound is None or upper_bound is None:
        return False
    lower_value, lower_inclusive = lower_bound
    upper_value, upper_inclusive = upper_bound
    if not isinstance(lower_value, bool | str) and not isinstance(
        upper_value, bool | str
    ):
        if lower_value > upper_value:
            return True
        if lower_value == upper_value and not (lower_inclusive and upper_inclusive):
            return True
    return False


def _affine_predicates_are_disjoint(
    predicates: tuple[StructuredPredicate, ...],
) -> bool:
    groups: dict[
        tuple[tuple[tuple[str, float], ...], float],
        list[AffineComparisonPredicate],
    ] = {}
    for predicate in predicates:
        if isinstance(predicate, AffineComparisonPredicate):
            key = (
                tuple(
                    sorted(
                        (
                            (_field_label(field), coefficient)
                            for field, coefficient in predicate.terms.items()
                        )
                    )
                ),
                predicate.offset,
            )
            groups.setdefault(key, []).append(predicate)
    return any(_affine_group_is_disjoint(tuple(group)) for group in groups.values())


def _affine_group_is_disjoint(
    predicates: tuple[AffineComparisonPredicate, ...],
) -> bool:
    lower_bound = _strongest_affine_lower_bound(predicates)
    upper_bound = _strongest_affine_upper_bound(predicates)
    if lower_bound is None or upper_bound is None:
        return False
    lower_value, lower_inclusive = lower_bound
    upper_value, upper_inclusive = upper_bound
    if lower_value > upper_value:
        return True
    return lower_value == upper_value and not (lower_inclusive and upper_inclusive)


def _strongest_affine_lower_bound(
    predicates: tuple[AffineComparisonPredicate, ...],
) -> tuple[float, bool] | None:
    lower: tuple[float, bool] | None = None
    for predicate in predicates:
        if predicate.operator not in {">", ">="}:
            continue
        candidate = (predicate.value, predicate.operator == ">=")
        if lower is None:
            lower = candidate
            continue
        candidate_value, candidate_inclusive = candidate
        lower_value, lower_inclusive = lower
        if candidate_value > lower_value:
            lower = candidate
        elif candidate_value == lower_value:
            lower = (candidate_value, candidate_inclusive and lower_inclusive)
    return lower


def _strongest_affine_upper_bound(
    predicates: tuple[AffineComparisonPredicate, ...],
) -> tuple[float, bool] | None:
    upper: tuple[float, bool] | None = None
    for predicate in predicates:
        if predicate.operator not in {"<", "<="}:
            continue
        candidate = (predicate.value, predicate.operator == "<=")
        if upper is None:
            upper = candidate
            continue
        candidate_value, candidate_inclusive = candidate
        upper_value, upper_inclusive = upper
        if candidate_value < upper_value:
            upper = candidate
        elif candidate_value == upper_value:
            upper = (candidate_value, candidate_inclusive and upper_inclusive)
    return upper


def _value_satisfies_comparisons(
    field: DescriptorField,
    value: ScalarValue,
    comparisons: tuple[ComparisonPredicate, ...],
) -> bool:
    descriptor = ParameterDescriptor({field: DescriptorCoordinate(value)})
    return all(predicate.evaluate(descriptor) for predicate in comparisons)


def _strongest_lower_bound(
    comparisons: tuple[ComparisonPredicate, ...],
) -> tuple[ScalarValue, bool] | None:
    lower: tuple[ScalarValue, bool] | None = None
    for predicate in comparisons:
        if predicate.operator not in {">", ">="}:
            continue
        candidate = (predicate.value, predicate.operator == ">=")
        candidate_value, candidate_inclusive = candidate
        if lower is None:
            lower = candidate
            continue
        lower_value, lower_inclusive = lower
        if candidate_value > lower_value:  # type: ignore[operator]
            lower = candidate
        elif candidate_value == lower_value:
            lower = (candidate_value, candidate_inclusive and lower_inclusive)
    return lower


def _strongest_upper_bound(
    comparisons: tuple[ComparisonPredicate, ...],
) -> tuple[ScalarValue, bool] | None:
    upper: tuple[ScalarValue, bool] | None = None
    for predicate in comparisons:
        if predicate.operator not in {"<", "<="}:
            continue
        candidate = (predicate.value, predicate.operator == "<=")
        candidate_value, candidate_inclusive = candidate
        if upper is None:
            upper = candidate
            continue
        upper_value, upper_inclusive = upper
        if candidate_value < upper_value:  # type: ignore[operator]
            upper = candidate
        elif candidate_value == upper_value:
            upper = (candidate_value, candidate_inclusive and upper_inclusive)
    return upper


@dataclass(frozen=True)
class ParameterSpaceSchema:
    """Parameter-space axes, validity rules, and coverage validation."""

    name: str
    axes: tuple[ParameterAxis, ...]
    auxiliary_fields: frozenset[DescriptorField] = frozenset()
    derived_regions: tuple[DerivedParameterRegion, ...] = ()
    invalid_cells: tuple[InvalidCellRule, ...] = ()

    @property
    def descriptor_fields(self) -> frozenset[DescriptorField]:
        """Descriptor fields declared by this schema."""
        return frozenset(axis.field for axis in self.axes) | self.auxiliary_fields

    def validate_schema(self) -> None:
        """Raise if the schema declaration is internally inconsistent."""
        fields = [axis.field for axis in self.axes]
        duplicates = sorted(
            (field for field in set(fields) if fields.count(field) > 1),
            key=_field_label,
        )
        if duplicates:
            raise ValueError(f"duplicate descriptor fields: {duplicates}")
        axis_fields = frozenset(fields)
        duplicate_auxiliary = sorted(
            axis_fields & self.auxiliary_fields,
            key=_field_label,
        )
        if duplicate_auxiliary:
            raise ValueError(
                "auxiliary descriptor fields duplicate axes: " f"{duplicate_auxiliary}"
            )
        empty_axes = [axis.label for axis in self.axes if not axis.bins]
        if empty_axes:
            raise ValueError(f"axes without bins or intervals: {empty_axes}")
        self.validate_derived_regions()
        self.validate_invalid_cells()

    def validate_descriptor(self, descriptor: ParameterDescriptor) -> None:
        """Raise if ``descriptor`` does not match this schema."""
        self.validate_schema()
        unknown_fields = set(descriptor.coordinates) - self.descriptor_fields
        if unknown_fields:
            raise ValueError(
                "descriptor has undeclared fields: "
                f"{sorted(map(_field_label, unknown_fields))}"
            )
        for axis in self.axes:
            coordinate = descriptor.coordinate(axis.field)
            if coordinate.known and not axis.contains(coordinate):
                raise ValueError(
                    "descriptor field "
                    f"{_field_label(axis.field)!r} is outside declared axis bins"
                )

    def validate_coverage_region(self, region: CoverageRegion) -> None:
        """Raise if ``region`` is not expressed over this schema."""
        self.validate_schema()
        for predicate in region.predicates:
            self._validate_predicate(predicate)
        unknown_fields = region.referenced_fields - self.descriptor_fields
        if unknown_fields:
            raise ValueError(
                f"coverage region {region.owner!r} references undeclared fields: "
                f"{sorted(map(_field_label, unknown_fields))}"
            )

    def validate_invalid_cells(self) -> None:
        """Raise if invalid-cell rules are not expressed over this schema."""
        for rule in self.invalid_cells:
            for predicate in rule.predicates:
                self._validate_predicate(predicate)
            unknown_fields = rule.referenced_fields - self.descriptor_fields
            if unknown_fields:
                raise ValueError(
                    f"invalid cell {rule.name!r} references undeclared fields: "
                    f"{sorted(map(_field_label, unknown_fields))}"
                )
            if not rule.reason:
                raise ValueError(f"invalid cell {rule.name!r} has no reason")

    def validate_derived_regions(self) -> None:
        """Raise if named regions are not expressed over this schema."""
        for region in self.derived_regions:
            if not region.alternatives:
                raise ValueError(f"derived region {region.name!r} has no alternatives")
            for alternative in region.alternatives:
                if not alternative:
                    raise ValueError(
                        f"derived region {region.name!r} has an empty alternative"
                    )
                for predicate in alternative:
                    self._validate_predicate(predicate)
            unknown_fields = region.referenced_fields - self.descriptor_fields
            if unknown_fields:
                raise ValueError(
                    f"derived region {region.name!r} references undeclared fields: "
                    f"{sorted(map(_field_label, unknown_fields))}"
                )

    def cell_status(
        self,
        descriptor: ParameterDescriptor,
        regions: tuple[CoverageRegion, ...],
    ) -> CellStatus:
        """Classify ``descriptor`` as invalid, owned, or uncovered."""
        self.validate_descriptor(descriptor)
        for region in regions:
            self.validate_coverage_region(region)
        if any(rule.matches(descriptor) for rule in self.invalid_cells):
            return "invalid"
        covering = self.covering_region(descriptor, regions)
        if covering is None:
            return "uncovered"
        return "owned"

    def covering_region(
        self,
        descriptor: ParameterDescriptor,
        regions: tuple[CoverageRegion, ...],
    ) -> CoverageRegion | None:
        """Return the unique coverage region containing ``descriptor``."""
        self.validate_descriptor(descriptor)
        for region in regions:
            self.validate_coverage_region(region)
        if any(rule.matches(descriptor) for rule in self.invalid_cells):
            raise ValueError(f"invalid descriptor {descriptor!r}")
        containing = tuple(region for region in regions if region.contains(descriptor))
        if not containing:
            return None
        if len(containing) > 1:
            owners = ", ".join(repr(region.owner) for region in containing)
            raise ValueError(f"descriptor lies in multiple coverage regions: {owners}")
        return containing[0]

    @staticmethod
    def _validate_predicate(predicate: StructuredPredicate) -> None:
        allowed = (
            ComparisonPredicate,
            MembershipPredicate,
            AffineComparisonPredicate,
            EvidencePredicate,
        )
        if not isinstance(predicate, allowed):
            raise TypeError(f"unsupported parameter-space predicate {predicate!r}")


_LINEARITY_EPS = 1.0e-12
_NUMERIC_EPS = 1.0e-14
LINEARITY_TOLERANCE = _LINEARITY_EPS
CONDITION_LIMIT = 1.0e8


def _axis(
    field: DescriptorField,
    bins: tuple[ParameterBin | NumericInterval, ...],
    *,
    units: str | None = None,
) -> ParameterAxis:
    return ParameterAxis(field, bins, units=units)


def _bool_axis(field: DescriptorField) -> ParameterAxis:
    return ParameterAxis(
        field,
        (
            ParameterBin("false", frozenset({False})),
            ParameterBin("true", frozenset({True})),
        ),
    )


def _nonnegative_axis(
    field: DescriptorField,
    *,
    units: str | None = None,
) -> ParameterAxis:
    return _axis(field, (NumericInterval("nonnegative", lower=0.0),), units=units)


def _positive_axis(
    field: DescriptorField,
    *,
    units: str | None = None,
) -> ParameterAxis:
    return _axis(
        field,
        (NumericInterval("positive", lower=0.0, include_lower=False),),
        units=units,
    )


def _defect_axis(field: DescriptorField) -> ParameterAxis:
    return _axis(
        field,
        (
            NumericInterval(
                "zero_to_linear_tolerance", lower=0.0, upper=_LINEARITY_EPS
            ),
            NumericInterval(
                "above_linear_tolerance",
                lower=_LINEARITY_EPS,
                include_lower=False,
            ),
        ),
        units="relative Frobenius norm unless field metadata says otherwise",
    )


def _solve_relation_axes() -> tuple[ParameterAxis, ...]:
    field = SolveRelationField
    return (
        _positive_axis(field.DIM_X, units="scalar unknowns"),
        _positive_axis(field.DIM_Y, units="scalar residual or target components"),
        _nonnegative_axis(field.AUXILIARY_SCALAR_COUNT, units="scalar unknowns"),
        _nonnegative_axis(field.EQUALITY_CONSTRAINT_COUNT, units="constraints"),
        _nonnegative_axis(field.NORMALIZATION_CONSTRAINT_COUNT, units="constraints"),
        _bool_axis(field.RESIDUAL_TARGET_AVAILABLE),
        _bool_axis(field.TARGET_IS_ZERO),
        _defect_axis(field.MAP_LINEARITY_DEFECT),
        _bool_axis(field.MATRIX_REPRESENTATION_AVAILABLE),
        _bool_axis(field.OPERATOR_APPLICATION_AVAILABLE),
        _axis(
            field.DERIVATIVE_ORACLE_KIND,
            (
                ParameterBin(
                    "oracle_kind",
                    frozenset({"none", "matrix", "jvp", "vjp", "jacobian_callback"}),
                ),
            ),
        ),
        _axis(
            field.OBJECTIVE_RELATION,
            (
                ParameterBin(
                    "objective_relation",
                    frozenset(
                        {
                            "none",
                            "residual_norm",
                            "least_squares",
                            "spectral_residual",
                        }
                    ),
                ),
            ),
        ),
        _axis(
            field.ACCEPTANCE_RELATION,
            (
                ParameterBin(
                    "acceptance_relation",
                    frozenset(
                        {
                            "residual_below_tolerance",
                            "objective_minimum",
                            "stationary_point",
                            "eigenpair_residual",
                        }
                    ),
                ),
            ),
        ),
        _positive_axis(field.REQUESTED_RESIDUAL_TOLERANCE, units="residual norm"),
        _positive_axis(field.REQUESTED_SOLUTION_TOLERANCE, units="solution norm"),
        _axis(
            field.BACKEND_KIND,
            (
                ParameterBin(
                    "backend_kind",
                    frozenset({"python", "numpy", "jax", "unknown"}),
                ),
            ),
        ),
        _axis(
            field.DEVICE_KIND,
            (ParameterBin("device_kind", frozenset({"cpu", "gpu", "unknown"})),),
        ),
        _positive_axis(field.WORK_BUDGET_FMAS, units="fused multiply-adds"),
        _positive_axis(field.MEMORY_BUDGET_BYTES, units="bytes"),
    )


def _linear_operator_axes() -> tuple[ParameterAxis, ...]:
    field = LinearSolverField
    return (
        _bool_axis(field.LINEAR_OPERATOR_MATRIX_AVAILABLE),
        _positive_axis(field.ASSEMBLY_COST_FMAS, units="fused multiply-adds"),
        _positive_axis(field.MATVEC_COST_FMAS, units="fused multiply-adds"),
        _positive_axis(field.LINEAR_OPERATOR_MEMORY_BYTES, units="bytes"),
        _axis(
            field.SYMMETRY_DEFECT,
            (
                NumericInterval("symmetric", lower=0.0, upper=_LINEARITY_EPS),
                NumericInterval(
                    "not_certified_symmetric",
                    lower=_LINEARITY_EPS,
                    include_lower=False,
                ),
            ),
            units="||A - A.T||_F / max(||A||_F, eps)",
        ),
        _axis(
            field.SKEW_SYMMETRY_DEFECT,
            (
                NumericInterval("skew_symmetric", lower=0.0, upper=_LINEARITY_EPS),
                NumericInterval(
                    "not_certified_skew_symmetric",
                    lower=_LINEARITY_EPS,
                    include_lower=False,
                ),
            ),
            units="||A + A.T||_F / max(||A||_F, eps)",
        ),
        _axis(
            field.DIAGONAL_NONZERO_MARGIN,
            (
                NumericInterval("zero_or_uncertified", upper=0.0),
                NumericInterval("nonzero", lower=0.0, include_lower=False),
            ),
        ),
        _axis(
            field.DIAGONAL_DOMINANCE_MARGIN,
            (
                NumericInterval("not_strict", upper=0.0),
                NumericInterval("strict", lower=0.0, include_lower=False),
            ),
        ),
        _axis(
            field.COERCIVITY_LOWER_BOUND,
            (
                NumericInterval("nonpositive", upper=0.0),
                NumericInterval("positive", lower=0.0, include_lower=False),
            ),
        ),
        _axis(
            field.SINGULAR_VALUE_LOWER_BOUND,
            (
                NumericInterval("zero_or_uncertified", upper=0.0),
                NumericInterval("positive", lower=0.0, include_lower=False),
            ),
        ),
        _axis(
            field.CONDITION_ESTIMATE,
            (
                NumericInterval("well_conditioned", lower=1.0, upper=1.0e8),
                NumericInterval("ill_conditioned", lower=1.0e8, include_lower=False),
            ),
        ),
        _nonnegative_axis(field.RANK_ESTIMATE, units="matrix rank"),
        _nonnegative_axis(field.NULLITY_ESTIMATE, units="matrix nullity"),
        _axis(
            field.RHS_CONSISTENCY_DEFECT,
            (
                NumericInterval("consistent", lower=0.0, upper=_LINEARITY_EPS),
                NumericInterval(
                    "inconsistent",
                    lower=_LINEARITY_EPS,
                    include_lower=False,
                ),
            ),
            units="||b - A x||_2 / max(||b||_2, eps)",
        ),
    )


def _solve_relation_regions() -> tuple[DerivedParameterRegion, ...]:
    field = SolveRelationField
    return (
        DerivedParameterRegion(
            "linear_system",
            (
                (
                    ComparisonPredicate(
                        field.MAP_LINEARITY_DEFECT,
                        "<=",
                        _LINEARITY_EPS,
                    ),
                    AffineComparisonPredicate(
                        {field.DIM_X: 1.0, field.DIM_Y: -1.0}, "==", 0.0
                    ),
                    MembershipPredicate(
                        field.RESIDUAL_TARGET_AVAILABLE, frozenset({True})
                    ),
                    MembershipPredicate(
                        field.ACCEPTANCE_RELATION,
                        frozenset({"residual_below_tolerance"}),
                    ),
                ),
            ),
        ),
        DerivedParameterRegion(
            "least_squares",
            (
                (
                    ComparisonPredicate(
                        field.MAP_LINEARITY_DEFECT,
                        "<=",
                        _LINEARITY_EPS,
                    ),
                    MembershipPredicate(
                        field.OBJECTIVE_RELATION, frozenset({"least_squares"})
                    ),
                    MembershipPredicate(
                        field.RESIDUAL_TARGET_AVAILABLE, frozenset({True})
                    ),
                ),
            ),
        ),
        DerivedParameterRegion(
            "nonlinear_root",
            (
                (
                    ComparisonPredicate(
                        field.MAP_LINEARITY_DEFECT, ">", _LINEARITY_EPS
                    ),
                    MembershipPredicate(
                        field.ACCEPTANCE_RELATION,
                        frozenset({"residual_below_tolerance"}),
                    ),
                ),
                (
                    EvidencePredicate(
                        field.MAP_LINEARITY_DEFECT, frozenset({"unavailable"})
                    ),
                    MembershipPredicate(
                        field.ACCEPTANCE_RELATION,
                        frozenset({"residual_below_tolerance"}),
                    ),
                ),
                (
                    MembershipPredicate(field.TARGET_IS_ZERO, frozenset({True})),
                    MembershipPredicate(
                        field.ACCEPTANCE_RELATION,
                        frozenset({"residual_below_tolerance"}),
                    ),
                ),
            ),
        ),
        DerivedParameterRegion(
            "eigenproblem",
            (
                (
                    ComparisonPredicate(field.AUXILIARY_SCALAR_COUNT, ">=", 1),
                    ComparisonPredicate(field.NORMALIZATION_CONSTRAINT_COUNT, ">=", 1),
                    MembershipPredicate(
                        field.ACCEPTANCE_RELATION,
                        frozenset({"eigenpair_residual"}),
                    ),
                ),
            ),
        ),
    )


def solve_relation_parameter_schema() -> ParameterSpaceSchema:
    """Return the primitive solve-relation parameter-space schema."""
    field = SolveRelationField
    return ParameterSpaceSchema(
        name="solve_relation",
        axes=_solve_relation_axes(),
        derived_regions=_solve_relation_regions(),
        invalid_cells=(
            InvalidCellRule(
                "eigenpair_requires_normalization",
                (
                    MembershipPredicate(
                        field.ACCEPTANCE_RELATION,
                        frozenset({"eigenpair_residual"}),
                    ),
                    ComparisonPredicate(field.NORMALIZATION_CONSTRAINT_COUNT, "==", 0),
                ),
                "eigenpair residuals require a normalization constraint",
            ),
            InvalidCellRule(
                "eigenpair_requires_auxiliary_scalar",
                (
                    MembershipPredicate(
                        field.ACCEPTANCE_RELATION,
                        frozenset({"eigenpair_residual"}),
                    ),
                    ComparisonPredicate(field.AUXILIARY_SCALAR_COUNT, "==", 0),
                ),
                "eigenpair residuals require a spectral auxiliary scalar",
            ),
        ),
    )


def linear_solver_parameter_schema() -> ParameterSpaceSchema:
    """Return the solve-relation schema extended with operator/decomposition axes."""
    field = LinearSolverField
    solve_field = SolveRelationField
    return ParameterSpaceSchema(
        name="linear_solver",
        axes=_solve_relation_axes() + _linear_operator_axes(),
        auxiliary_fields=frozenset(DecompositionField),
        derived_regions=_solve_relation_regions()
        + (
            DerivedParameterRegion(
                "square",
                (
                    (
                        AffineComparisonPredicate(
                            {solve_field.DIM_X: 1.0, solve_field.DIM_Y: -1.0},
                            "==",
                            0.0,
                        ),
                    ),
                ),
            ),
            DerivedParameterRegion(
                "overdetermined",
                (
                    (
                        AffineComparisonPredicate(
                            {solve_field.DIM_Y: 1.0, solve_field.DIM_X: -1.0},
                            ">",
                            0.0,
                        ),
                    ),
                ),
            ),
            DerivedParameterRegion(
                "full_rank",
                ((ComparisonPredicate(field.SINGULAR_VALUE_LOWER_BOUND, ">", 0.0),),),
            ),
            DerivedParameterRegion(
                "rank_deficient",
                ((ComparisonPredicate(field.NULLITY_ESTIMATE, ">", 0),),),
            ),
            DerivedParameterRegion(
                "symmetric_positive_definite",
                (
                    (
                        AffineComparisonPredicate(
                            {solve_field.DIM_X: 1.0, solve_field.DIM_Y: -1.0},
                            "==",
                            0.0,
                        ),
                        ComparisonPredicate(
                            field.SYMMETRY_DEFECT, "<=", _LINEARITY_EPS
                        ),
                        ComparisonPredicate(field.COERCIVITY_LOWER_BOUND, ">", 0.0),
                    ),
                ),
            ),
            DerivedParameterRegion(
                "matrix_free",
                (
                    (
                        MembershipPredicate(
                            field.LINEAR_OPERATOR_MATRIX_AVAILABLE,
                            frozenset({False}),
                        ),
                        MembershipPredicate(
                            solve_field.OPERATOR_APPLICATION_AVAILABLE,
                            frozenset({True}),
                        ),
                    ),
                ),
            ),
        ),
        invalid_cells=(
            InvalidCellRule(
                "coercivity_requires_square_map",
                (
                    AffineComparisonPredicate(
                        {solve_field.DIM_X: 1.0, solve_field.DIM_Y: -1.0}, "!=", 0.0
                    ),
                    ComparisonPredicate(field.COERCIVITY_LOWER_BOUND, ">", 0.0),
                ),
                "positive coercivity is meaningful only for square maps",
            ),
            InvalidCellRule(
                "symmetry_requires_square_map",
                (
                    AffineComparisonPredicate(
                        {solve_field.DIM_X: 1.0, solve_field.DIM_Y: -1.0}, "!=", 0.0
                    ),
                    ComparisonPredicate(field.SYMMETRY_DEFECT, "<=", _LINEARITY_EPS),
                ),
                "matrix symmetry is meaningful only for square maps",
            ),
        ),
    )


def decomposition_parameter_schema() -> ParameterSpaceSchema:
    """Return the dense-matrix decomposition parameter-space schema."""
    return ParameterSpaceSchema(
        name="decomposition",
        axes=_decomposition_axes(),
        derived_regions=_decomposition_regions(),
    )


def reaction_network_parameter_schema() -> ParameterSpaceSchema:
    """Return the stoichiometric reaction-network parameter-space schema."""
    field = ReactionNetworkField
    return ParameterSpaceSchema(
        name="reaction_network",
        axes=(
            _positive_axis(field.SPECIES_COUNT, units="species"),
            _nonnegative_axis(field.REACTION_COUNT, units="reaction pairs"),
            _nonnegative_axis(field.STOICHIOMETRY_RANK, units="matrix rank"),
            _nonnegative_axis(
                field.CONSERVATION_LAW_COUNT, units="independent linear forms"
            ),
            _nonnegative_axis(
                field.EQUILIBRIUM_CONSTRAINT_COUNT,
                units="independent equilibrium conditions",
            ),
        ),
        derived_regions=(
            DerivedParameterRegion(
                "conserved_network",
                ((ComparisonPredicate(field.CONSERVATION_LAW_COUNT, ">", 0),),),
            ),
            DerivedParameterRegion(
                "fully_constrained_equilibrium",
                (
                    (
                        AffineComparisonPredicate(
                            {
                                field.EQUILIBRIUM_CONSTRAINT_COUNT: 1.0,
                                field.STOICHIOMETRY_RANK: -1.0,
                            },
                            "==",
                            0.0,
                        ),
                    ),
                ),
            ),
        ),
        invalid_cells=(
            InvalidCellRule(
                "rank_exceeds_species_count",
                (
                    AffineComparisonPredicate(
                        {
                            field.STOICHIOMETRY_RANK: 1.0,
                            field.SPECIES_COUNT: -1.0,
                        },
                        ">",
                        0.0,
                    ),
                ),
                "stoichiometry rank cannot exceed species count",
            ),
            InvalidCellRule(
                "rank_exceeds_reaction_count",
                (
                    AffineComparisonPredicate(
                        {
                            field.STOICHIOMETRY_RANK: 1.0,
                            field.REACTION_COUNT: -1.0,
                        },
                        ">",
                        0.0,
                    ),
                ),
                "stoichiometry rank cannot exceed reaction count",
            ),
            InvalidCellRule(
                "rank_plus_conservation_must_equal_species_count",
                (
                    AffineComparisonPredicate(
                        {
                            field.STOICHIOMETRY_RANK: 1.0,
                            field.CONSERVATION_LAW_COUNT: 1.0,
                            field.SPECIES_COUNT: -1.0,
                        },
                        "!=",
                        0.0,
                    ),
                ),
                "left-nullity plus rank must equal the number of species",
            ),
            InvalidCellRule(
                "equilibrium_constraints_exceed_stoichiometry_rank",
                (
                    AffineComparisonPredicate(
                        {
                            field.EQUILIBRIUM_CONSTRAINT_COUNT: 1.0,
                            field.STOICHIOMETRY_RANK: -1.0,
                        },
                        ">",
                        0.0,
                    ),
                ),
                "independent equilibrium constraints cannot exceed stoichiometry rank",
            ),
        ),
    )


def map_structure_parameter_schema() -> ParameterSpaceSchema:
    """Return the ODE map/state-memory parameter-space schema."""
    field = MapStructureField
    return ParameterSpaceSchema(
        name="map_structure",
        axes=(
            _bool_axis(field.RHS_EVALUATION_AVAILABLE),
            _bool_axis(field.RHS_HISTORY_AVAILABLE),
            _bool_axis(field.NORDSIECK_HISTORY_AVAILABLE),
            _bool_axis(field.EXACT_LINEAR_OPERATOR_AVAILABLE),
            _bool_axis(field.NONLINEAR_RESIDUAL_AVAILABLE),
            _bool_axis(field.EXPLICIT_COMPONENT_AVAILABLE),
            _bool_axis(field.IMPLICIT_COMPONENT_AVAILABLE),
            ParameterAxis(
                field.IMPLICIT_COMPONENT_DERIVATIVE_ORACLE_KIND,
                (
                    ParameterBin(
                        "implicit_component_derivative_oracle_kind",
                        frozenset({"jacobian_callback", "matrix", "unavailable"}),
                    ),
                ),
            ),
            _bool_axis(field.HAMILTONIAN_PARTITION_AVAILABLE),
            _bool_axis(field.SYMPLECTIC_FORM_INVARIANT_AVAILABLE),
            _nonnegative_axis(field.ADDITIVE_COMPONENT_COUNT, units="components"),
        ),
        derived_regions=(
            DerivedParameterRegion(
                "single_step_rhs_evaluation",
                (
                    (
                        MembershipPredicate(
                            field.RHS_EVALUATION_AVAILABLE, frozenset({True})
                        ),
                        MembershipPredicate(
                            field.RHS_HISTORY_AVAILABLE, frozenset({False})
                        ),
                        MembershipPredicate(
                            field.NORDSIECK_HISTORY_AVAILABLE, frozenset({False})
                        ),
                    ),
                ),
            ),
            DerivedParameterRegion(
                "rhs_history_state",
                (
                    (
                        MembershipPredicate(
                            field.RHS_EVALUATION_AVAILABLE, frozenset({True})
                        ),
                        MembershipPredicate(
                            field.RHS_HISTORY_AVAILABLE, frozenset({True})
                        ),
                    ),
                ),
            ),
            DerivedParameterRegion(
                "nordsieck_state",
                (
                    (
                        MembershipPredicate(
                            field.NORDSIECK_HISTORY_AVAILABLE, frozenset({True})
                        ),
                    ),
                ),
            ),
        ),
    )


def _decomposition_axes() -> tuple[ParameterAxis, ...]:
    decomposition_field = DecompositionField
    return (
        _positive_axis(decomposition_field.MATRIX_ROWS, units="rows"),
        _positive_axis(decomposition_field.MATRIX_COLUMNS, units="columns"),
        _positive_axis(
            decomposition_field.FACTORIZATION_WORK_BUDGET_FMAS,
            units="fused multiply-adds",
        ),
        _positive_axis(
            decomposition_field.FACTORIZATION_MEMORY_BUDGET_BYTES, units="bytes"
        ),
        _axis(
            decomposition_field.SINGULAR_VALUE_LOWER_BOUND,
            (
                NumericInterval("zero_or_uncertified", upper=0.0),
                NumericInterval("positive", lower=0.0, include_lower=False),
            ),
        ),
        _axis(
            decomposition_field.CONDITION_ESTIMATE,
            (
                NumericInterval("well_conditioned", lower=1.0, upper=1.0e8),
                NumericInterval("ill_conditioned", lower=1.0e8, include_lower=False),
            ),
        ),
        _nonnegative_axis(decomposition_field.MATRIX_RANK_ESTIMATE),
        _nonnegative_axis(decomposition_field.MATRIX_NULLITY_ESTIMATE),
    )


def _decomposition_regions() -> tuple[DerivedParameterRegion, ...]:
    decomposition_field = DecompositionField
    return (
        DerivedParameterRegion(
            "square",
            (
                (
                    AffineComparisonPredicate(
                        {
                            decomposition_field.MATRIX_ROWS: 1.0,
                            decomposition_field.MATRIX_COLUMNS: -1.0,
                        },
                        "==",
                        0.0,
                    ),
                ),
            ),
        ),
        DerivedParameterRegion(
            "full_rank",
            (
                (
                    ComparisonPredicate(
                        decomposition_field.SINGULAR_VALUE_LOWER_BOUND, ">", 0.0
                    ),
                ),
            ),
        ),
        DerivedParameterRegion(
            "rank_deficient",
            (
                (
                    ComparisonPredicate(
                        decomposition_field.MATRIX_NULLITY_ESTIMATE, ">", 0
                    ),
                ),
            ),
        ),
    )


def decomposition_descriptor_from_linear_operator_descriptor(
    descriptor: LinearOperatorDescriptor,
    *,
    factorization_work_budget_fmas: float = 1.0e9,
    factorization_memory_budget_bytes: float = 1.0e9,
) -> ParameterDescriptor:
    """Project a linear-operator descriptor onto dense decomposition coordinates."""
    source = descriptor.parameter_descriptor
    return ParameterDescriptor(
        {
            DecompositionField.MATRIX_ROWS: DescriptorCoordinate(
                len(descriptor.matrix)
            ),
            DecompositionField.MATRIX_COLUMNS: DescriptorCoordinate(
                len(descriptor.matrix[0]) if descriptor.matrix else 0
            ),
            DecompositionField.FACTORIZATION_WORK_BUDGET_FMAS: DescriptorCoordinate(
                factorization_work_budget_fmas
            ),
            DecompositionField.FACTORIZATION_MEMORY_BUDGET_BYTES: DescriptorCoordinate(
                factorization_memory_budget_bytes
            ),
            DecompositionField.SINGULAR_VALUE_LOWER_BOUND: source.coordinate(
                LinearSolverField.SINGULAR_VALUE_LOWER_BOUND
            ),
            DecompositionField.CONDITION_ESTIMATE: source.coordinate(
                LinearSolverField.CONDITION_ESTIMATE
            ),
            DecompositionField.MATRIX_RANK_ESTIMATE: source.coordinate(
                LinearSolverField.RANK_ESTIMATE
            ),
            DecompositionField.MATRIX_NULLITY_ESTIMATE: source.coordinate(
                LinearSolverField.NULLITY_ESTIMATE
            ),
        }
    )


def _backend_kind(backend: object) -> str:
    name = type(backend).__name__.lower()
    if "numpy" in name:
        return "numpy"
    if "jax" in name:
        return "jax"
    if "python" in name:
        return "python"
    return "unknown"


def _assemble_matrix(
    op: SmallLinearOperator, b: Tensor
) -> tuple[tuple[float, ...], ...]:
    n = b.shape[0]
    columns: list[list[float]] = []
    for col in range(n):
        basis = Tensor.zeros(n, backend=b.backend)
        basis = basis.set(col, Tensor(1.0, backend=b.backend))
        applied = op.apply(basis)
        columns.append([float(applied[row]) for row in range(n)])
    return tuple(tuple(columns[col][row] for col in range(n)) for row in range(n))


def _rhs_tuple(b: Tensor) -> tuple[float, ...]:
    return tuple(float(b[row]) for row in range(b.shape[0]))


def _zero_small(value: float, tolerance: float = _NUMERIC_EPS) -> float:
    return 0.0 if abs(value) <= tolerance else value


def linear_operator_descriptor_from_assembled_operator(
    op: SmallLinearOperator,
    b: Tensor,
    *,
    requested_residual_tolerance: float = 1.0e-8,
    requested_solution_tolerance: float = 1.0e-8,
    work_budget_fmas: float = 1.0e9,
    memory_budget_bytes: float = 1.0e9,
    device_kind: str = "cpu",
) -> LinearOperatorDescriptor:
    """Assemble a small operator and return deterministic schema coordinates.

    The construction is for capability classification and structural fixtures,
    not for benchmarking.  It materializes the matrix by applying ``op`` to each
    standard basis vector, then computes algebraic descriptors using stable
    dense linear-algebra formulas.
    """
    if len(b.shape) != 1 or b.shape[0] <= 0:
        raise ValueError("linear-operator descriptors require a nonempty vector RHS")

    matrix = _assemble_matrix(op, b)
    rhs = _rhs_tuple(b)
    n = len(rhs)
    a = np.array(matrix, dtype=float)
    rhs_array = np.array(rhs, dtype=float)
    frobenius_norm = float(np.linalg.norm(a))
    scale = max(frobenius_norm, _NUMERIC_EPS)
    symmetry_defect = _zero_small(float(np.linalg.norm(a - a.T)) / scale)
    skew_symmetry_defect = _zero_small(float(np.linalg.norm(a + a.T)) / scale)
    diagonal_nonzero_margin = min(abs(matrix[i][i]) for i in range(n))
    diagonal_dominance_margin = min(
        abs(matrix[i][i]) - sum(abs(matrix[i][j]) for j in range(n) if j != i)
        for i in range(n)
    )

    singular_values = np.linalg.svd(a, compute_uv=False)
    singular_tolerance = (
        max(a.shape)
        * np.finfo(float).eps
        * max(
            float(singular_values[0]) if singular_values.size else 0.0,
            1.0,
        )
    )
    cleaned_singular_values = [
        0.0 if value <= singular_tolerance else float(value)
        for value in singular_values
    ]
    singular_value_lower_bound = min(cleaned_singular_values, default=0.0)
    rank_estimate = sum(value > 0.0 for value in cleaned_singular_values)
    nullity_estimate = n - rank_estimate
    positive_singular_values = [
        value for value in cleaned_singular_values if value > 0.0
    ]
    condition_estimate = (
        max(cleaned_singular_values) / min(positive_singular_values)
        if positive_singular_values
        else 1.0e300
    )

    symmetric_part = 0.5 * (a + a.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_part)
    coercivity_lower_bound = _zero_small(float(np.min(eigenvalues)))

    least_squares_solution, *_ = np.linalg.lstsq(a, rhs_array, rcond=None)
    residual = rhs_array - a @ least_squares_solution
    rhs_consistency_defect = _zero_small(
        float(np.linalg.norm(residual))
        / max(float(np.linalg.norm(rhs_array)), _NUMERIC_EPS)
    )
    nonzeros = int(np.count_nonzero(np.abs(a) > _NUMERIC_EPS))
    matvec_cost_fmas = float(max(1, 2 * nonzeros))
    assembly_cost_fmas = float(n) * matvec_cost_fmas
    memory_estimate = float(8 * (n * n + n))

    descriptor = ParameterDescriptor(
        {
            SolveRelationField.DIM_X: DescriptorCoordinate(n),
            SolveRelationField.DIM_Y: DescriptorCoordinate(n),
            SolveRelationField.AUXILIARY_SCALAR_COUNT: DescriptorCoordinate(0),
            SolveRelationField.EQUALITY_CONSTRAINT_COUNT: DescriptorCoordinate(0),
            SolveRelationField.NORMALIZATION_CONSTRAINT_COUNT: DescriptorCoordinate(0),
            SolveRelationField.RESIDUAL_TARGET_AVAILABLE: DescriptorCoordinate(True),
            SolveRelationField.TARGET_IS_ZERO: DescriptorCoordinate(False),
            SolveRelationField.MAP_LINEARITY_DEFECT: DescriptorCoordinate(0.0),
            SolveRelationField.MATRIX_REPRESENTATION_AVAILABLE: DescriptorCoordinate(
                True
            ),
            SolveRelationField.OPERATOR_APPLICATION_AVAILABLE: DescriptorCoordinate(
                True
            ),
            SolveRelationField.DERIVATIVE_ORACLE_KIND: DescriptorCoordinate("matrix"),
            SolveRelationField.OBJECTIVE_RELATION: DescriptorCoordinate("none"),
            SolveRelationField.ACCEPTANCE_RELATION: DescriptorCoordinate(
                "residual_below_tolerance"
            ),
            SolveRelationField.REQUESTED_RESIDUAL_TOLERANCE: DescriptorCoordinate(
                requested_residual_tolerance
            ),
            SolveRelationField.REQUESTED_SOLUTION_TOLERANCE: DescriptorCoordinate(
                requested_solution_tolerance
            ),
            SolveRelationField.BACKEND_KIND: DescriptorCoordinate(
                _backend_kind(b.backend)
            ),
            SolveRelationField.DEVICE_KIND: DescriptorCoordinate(device_kind),
            SolveRelationField.WORK_BUDGET_FMAS: DescriptorCoordinate(work_budget_fmas),
            SolveRelationField.MEMORY_BUDGET_BYTES: DescriptorCoordinate(
                memory_budget_bytes
            ),
            LinearSolverField.LINEAR_OPERATOR_MATRIX_AVAILABLE: DescriptorCoordinate(
                True
            ),
            LinearSolverField.ASSEMBLY_COST_FMAS: DescriptorCoordinate(
                assembly_cost_fmas
            ),
            LinearSolverField.MATVEC_COST_FMAS: DescriptorCoordinate(matvec_cost_fmas),
            LinearSolverField.LINEAR_OPERATOR_MEMORY_BYTES: DescriptorCoordinate(
                memory_estimate
            ),
            LinearSolverField.SYMMETRY_DEFECT: DescriptorCoordinate(symmetry_defect),
            LinearSolverField.SKEW_SYMMETRY_DEFECT: DescriptorCoordinate(
                skew_symmetry_defect
            ),
            LinearSolverField.DIAGONAL_NONZERO_MARGIN: DescriptorCoordinate(
                diagonal_nonzero_margin
            ),
            LinearSolverField.DIAGONAL_DOMINANCE_MARGIN: DescriptorCoordinate(
                diagonal_dominance_margin
            ),
            LinearSolverField.COERCIVITY_LOWER_BOUND: DescriptorCoordinate(
                coercivity_lower_bound, evidence="lower_bound"
            ),
            LinearSolverField.SINGULAR_VALUE_LOWER_BOUND: DescriptorCoordinate(
                singular_value_lower_bound, evidence="lower_bound"
            ),
            LinearSolverField.CONDITION_ESTIMATE: DescriptorCoordinate(
                condition_estimate, evidence="estimate"
            ),
            LinearSolverField.RANK_ESTIMATE: DescriptorCoordinate(
                rank_estimate, evidence="estimate"
            ),
            LinearSolverField.NULLITY_ESTIMATE: DescriptorCoordinate(
                nullity_estimate, evidence="estimate"
            ),
            LinearSolverField.RHS_CONSISTENCY_DEFECT: DescriptorCoordinate(
                rhs_consistency_defect
            ),
            DecompositionField.MATRIX_ROWS: DescriptorCoordinate(n),
            DecompositionField.MATRIX_COLUMNS: DescriptorCoordinate(n),
            DecompositionField.FACTORIZATION_WORK_BUDGET_FMAS: DescriptorCoordinate(
                work_budget_fmas
            ),
            DecompositionField.FACTORIZATION_MEMORY_BUDGET_BYTES: (
                DescriptorCoordinate(memory_budget_bytes)
            ),
            DecompositionField.SINGULAR_VALUE_LOWER_BOUND: DescriptorCoordinate(
                singular_value_lower_bound, evidence="lower_bound"
            ),
            DecompositionField.CONDITION_ESTIMATE: DescriptorCoordinate(
                condition_estimate, evidence="estimate"
            ),
            DecompositionField.MATRIX_RANK_ESTIMATE: DescriptorCoordinate(
                rank_estimate, evidence="estimate"
            ),
            DecompositionField.MATRIX_NULLITY_ESTIMATE: DescriptorCoordinate(
                nullity_estimate, evidence="estimate"
            ),
        }
    )
    return LinearOperatorDescriptor(descriptor, matrix)


__all__ = [
    "AffineComparisonPredicate",
    "AlgorithmCapability",
    "AlgorithmRegistry",
    "AlgorithmRequest",
    "AlgorithmStructureContract",
    "ComparisonPredicate",
    "CONDITION_LIMIT",
    "coverage_regions_are_disjoint",
    "CoverageRegion",
    "DecompositionField",
    "DerivedParameterRegion",
    "DescriptorCoordinate",
    "decomposition_descriptor_from_linear_operator_descriptor",
    "decomposition_parameter_schema",
    "EvidencePredicate",
    "InvalidCellRule",
    "LinearSolverField",
    "LINEARITY_TOLERANCE",
    "LinearOperatorDescriptor",
    "linear_solver_parameter_schema",
    "linear_operator_descriptor_from_assembled_operator",
    "MapStructureField",
    "map_structure_parameter_schema",
    "MembershipPredicate",
    "NumericInterval",
    "ParameterAxis",
    "ParameterBin",
    "ParameterDescriptor",
    "ParameterPredicate",
    "ParameterSpaceSchema",
    "predicate_sets_are_disjoint",
    "ReactionNetworkField",
    "reaction_network_parameter_schema",
    "SmallLinearOperator",
    "solve_relation_parameter_schema",
    "SolveRelationField",
]
