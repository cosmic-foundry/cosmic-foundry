"""Shared algorithm capability declarations and selection."""

from __future__ import annotations

from dataclasses import dataclass
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
    priority: int | None = None
    coverage_patches: tuple[CoveragePatch, ...] = ()

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
        return (
            self.contract.requires <= request.available_structure
            and request.requested_properties <= self.contract.provides
        )


@dataclass(frozen=True)
class AlgorithmRequest:
    """Requested input structure and desired algorithmic properties."""

    available_structure: frozenset[str] = frozenset()
    requested_properties: frozenset[str] = frozenset()
    order: int | None = None


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
        """Return the unique or explicitly prioritized implementation."""
        matches = self.matching(request)
        if not matches:
            raise ValueError(f"no algorithm satisfies request {request!r}")
        if len(matches) == 1:
            return matches[0]

        ranked = [cap for cap in matches if cap.priority is not None]
        if not ranked:
            names = ", ".join(cap.name for cap in matches)
            raise ValueError(f"ambiguous algorithm request {request!r}: {names}")
        ranked.sort(key=lambda cap: cap.priority if cap.priority is not None else 0)
        if len(ranked) > 1 and ranked[0].priority == ranked[1].priority:
            names = ", ".join(cap.name for cap in ranked)
            raise ValueError(f"ambiguous algorithm priority {request!r}: {names}")
        return ranked[0]


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

    name: str
    bins: tuple[ParameterBin | NumericInterval, ...]
    descriptor_field: str | None = None
    units: str | None = None

    @property
    def field(self) -> str:
        """Descriptor field that locates a value on this axis."""
        return self.descriptor_field or self.name

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
    """Concrete problem location in a parameter-space schema."""

    schema: str
    coordinates: dict[str, DescriptorCoordinate]

    def coordinate(self, field: str) -> DescriptorCoordinate:
        """Return the coordinate for ``field`` or an explicit unavailable value."""
        return self.coordinates.get(
            field, DescriptorCoordinate(None, evidence="unavailable")
        )


class ParameterPredicate(Protocol):
    """Structured predicate over descriptor coordinates."""

    referenced_fields: frozenset[str]

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

    def coordinate(self, field: str) -> DescriptorCoordinate:
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

    field: str
    operator: ComparisonOperator
    value: ScalarValue
    accepted_evidence: frozenset[EvidenceSource] = frozenset(
        {"exact", "upper_bound", "lower_bound", "estimate"}
    )

    @property
    def referenced_fields(self) -> frozenset[str]:
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

    field: str
    values: frozenset[ScalarValue]
    accepted_evidence: frozenset[EvidenceSource] = frozenset(
        {"exact", "caller_assumption"}
    )

    @property
    def referenced_fields(self) -> frozenset[str]:
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

    terms: dict[str, float]
    operator: ComparisonOperator
    value: float
    offset: float = 0.0
    accepted_evidence: frozenset[EvidenceSource] = frozenset(
        {"exact", "upper_bound", "lower_bound", "estimate"}
    )

    @property
    def referenced_fields(self) -> frozenset[str]:
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

    field: str
    evidence: frozenset[EvidenceSource]

    @property
    def referenced_fields(self) -> frozenset[str]:
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
    def referenced_fields(self) -> frozenset[str]:
        """Descriptor fields referenced by this derived region."""
        fields: set[str] = set()
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
    def referenced_fields(self) -> frozenset[str]:
        """Descriptor fields referenced by this invalid-cell rule."""
        return frozenset().union(
            *(predicate.referenced_fields for predicate in self.predicates)
        )

    def matches(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether ``descriptor`` inhabits this invalid region."""
        return all(predicate.evaluate(descriptor) for predicate in self.predicates)


@dataclass(frozen=True)
class CoveragePatch:
    """Owned parameter-space region claimed by one implementation."""

    name: str
    owner: str
    predicates: tuple[StructuredPredicate, ...]

    @property
    def referenced_fields(self) -> frozenset[str]:
        """Descriptor fields referenced by this coverage patch."""
        return frozenset().union(
            *(predicate.referenced_fields for predicate in self.predicates)
        )

    def contains(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether ``descriptor`` lies inside this patch."""
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


def coverage_patches_are_disjoint(patches: tuple[CoveragePatch, ...]) -> bool:
    """Return whether coverage patches form a pairwise-disjoint partition."""
    for pair_index, left in enumerate(patches):
        for right in patches[pair_index + 1 :]:
            if not predicate_sets_are_disjoint(left.predicates, right.predicates):
                return False
    return True


def _field_predicates_are_disjoint(
    field: str,
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
    field: str,
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
            key = (tuple(sorted(predicate.terms.items())), predicate.offset)
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
    field: str,
    value: ScalarValue,
    comparisons: tuple[ComparisonPredicate, ...],
) -> bool:
    descriptor = ParameterDescriptor("witness", {field: DescriptorCoordinate(value)})
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
    derived_regions: tuple[DerivedParameterRegion, ...] = ()
    invalid_cells: tuple[InvalidCellRule, ...] = ()

    @property
    def descriptor_fields(self) -> frozenset[str]:
        """Descriptor fields declared by this schema."""
        return frozenset(axis.field for axis in self.axes)

    def validate_schema(self) -> None:
        """Raise if the schema declaration is internally inconsistent."""
        fields = [axis.field for axis in self.axes]
        duplicates = sorted(field for field in set(fields) if fields.count(field) > 1)
        if duplicates:
            raise ValueError(f"duplicate descriptor fields: {duplicates}")
        empty_axes = [axis.name for axis in self.axes if not axis.bins]
        if empty_axes:
            raise ValueError(f"axes without bins or intervals: {empty_axes}")
        self.validate_derived_regions()
        self.validate_invalid_cells()

    def validate_descriptor(self, descriptor: ParameterDescriptor) -> None:
        """Raise if ``descriptor`` does not match this schema."""
        self.validate_schema()
        if descriptor.schema != self.name:
            raise ValueError(
                f"descriptor schema {descriptor.schema!r} does not match {self.name!r}"
            )
        unknown_fields = set(descriptor.coordinates) - self.descriptor_fields
        if unknown_fields:
            raise ValueError(
                f"descriptor has undeclared fields: {sorted(unknown_fields)}"
            )
        for axis in self.axes:
            coordinate = descriptor.coordinate(axis.field)
            if coordinate.known and not axis.contains(coordinate):
                raise ValueError(
                    f"descriptor field {axis.field!r} is outside declared axis bins"
                )

    def validate_coverage_patch(self, patch: CoveragePatch) -> None:
        """Raise if ``patch`` is not expressed over this schema."""
        self.validate_schema()
        for predicate in patch.predicates:
            self._validate_predicate(predicate)
        unknown_fields = patch.referenced_fields - self.descriptor_fields
        if unknown_fields:
            raise ValueError(
                f"coverage patch {patch.name!r} references undeclared fields: "
                f"{sorted(unknown_fields)}"
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
                    f"{sorted(unknown_fields)}"
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
                    f"{sorted(unknown_fields)}"
                )

    def cell_status(
        self,
        descriptor: ParameterDescriptor,
        patches: tuple[CoveragePatch, ...],
    ) -> CellStatus:
        """Classify ``descriptor`` as invalid, owned, or uncovered."""
        self.validate_descriptor(descriptor)
        for patch in patches:
            self.validate_coverage_patch(patch)
        if any(rule.matches(descriptor) for rule in self.invalid_cells):
            return "invalid"
        covering = self.covering_patch(descriptor, patches)
        if covering is None:
            return "uncovered"
        return "owned"

    def covering_patch(
        self,
        descriptor: ParameterDescriptor,
        patches: tuple[CoveragePatch, ...],
    ) -> CoveragePatch | None:
        """Return the unique coverage patch containing ``descriptor``."""
        self.validate_descriptor(descriptor)
        for patch in patches:
            self.validate_coverage_patch(patch)
        if any(rule.matches(descriptor) for rule in self.invalid_cells):
            raise ValueError(f"invalid descriptor {descriptor!r}")
        containing = tuple(patch for patch in patches if patch.contains(descriptor))
        if not containing:
            return None
        if len(containing) > 1:
            names = ", ".join(patch.name for patch in containing)
            raise ValueError(f"descriptor lies in multiple coverage patches: {names}")
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


def _bool_axis(name: str) -> ParameterAxis:
    return ParameterAxis(
        name,
        (
            ParameterBin("false", frozenset({False})),
            ParameterBin("true", frozenset({True})),
        ),
    )


def _nonnegative_axis(name: str, *, units: str | None = None) -> ParameterAxis:
    return ParameterAxis(
        name, (NumericInterval("nonnegative", lower=0.0),), units=units
    )


def _positive_axis(name: str, *, units: str | None = None) -> ParameterAxis:
    return ParameterAxis(
        name,
        (NumericInterval("positive", lower=0.0, include_lower=False),),
        units=units,
    )


def _defect_axis(name: str) -> ParameterAxis:
    return ParameterAxis(
        name,
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
    return (
        _positive_axis("dim_x", units="scalar unknowns"),
        _positive_axis("dim_y", units="scalar residual or target components"),
        _nonnegative_axis("auxiliary_scalar_count", units="scalar unknowns"),
        _nonnegative_axis("equality_constraint_count", units="constraints"),
        _nonnegative_axis("normalization_constraint_count", units="constraints"),
        _bool_axis("residual_target_available"),
        _bool_axis("target_is_zero"),
        _defect_axis("map_linearity_defect"),
        _bool_axis("matrix_representation_available"),
        _bool_axis("operator_application_available"),
        ParameterAxis(
            "derivative_oracle_kind",
            (
                ParameterBin(
                    "oracle_kind",
                    frozenset({"none", "matrix", "jvp", "vjp", "jacobian_callback"}),
                ),
            ),
        ),
        ParameterAxis(
            "objective_relation",
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
        ParameterAxis(
            "acceptance_relation",
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
        _positive_axis("requested_residual_tolerance", units="residual norm"),
        _positive_axis("requested_solution_tolerance", units="solution norm"),
        ParameterAxis(
            "backend_kind",
            (
                ParameterBin(
                    "backend_kind",
                    frozenset({"python", "numpy", "jax", "unknown"}),
                ),
            ),
        ),
        ParameterAxis(
            "device_kind",
            (ParameterBin("device_kind", frozenset({"cpu", "gpu", "unknown"})),),
        ),
        _positive_axis("work_budget_fmas", units="fused multiply-adds"),
        _positive_axis("memory_budget_bytes", units="bytes"),
    )


def _linear_operator_axes() -> tuple[ParameterAxis, ...]:
    return (
        _bool_axis("linear_operator_matrix_available"),
        _positive_axis("assembly_cost_fmas", units="fused multiply-adds"),
        _positive_axis("matvec_cost_fmas", units="fused multiply-adds"),
        _positive_axis("linear_operator_memory_bytes", units="bytes"),
        ParameterAxis(
            "symmetry_defect",
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
        ParameterAxis(
            "skew_symmetry_defect",
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
        ParameterAxis(
            "diagonal_nonzero_margin",
            (
                NumericInterval("zero_or_uncertified", upper=0.0),
                NumericInterval("nonzero", lower=0.0, include_lower=False),
            ),
        ),
        ParameterAxis(
            "diagonal_dominance_margin",
            (
                NumericInterval("not_strict", upper=0.0),
                NumericInterval("strict", lower=0.0, include_lower=False),
            ),
        ),
        ParameterAxis(
            "coercivity_lower_bound",
            (
                NumericInterval("nonpositive", upper=0.0),
                NumericInterval("positive", lower=0.0, include_lower=False),
            ),
        ),
        ParameterAxis(
            "singular_value_lower_bound",
            (
                NumericInterval("zero_or_uncertified", upper=0.0),
                NumericInterval("positive", lower=0.0, include_lower=False),
            ),
        ),
        ParameterAxis(
            "condition_estimate",
            (
                NumericInterval("well_conditioned", lower=1.0, upper=1.0e8),
                NumericInterval("ill_conditioned", lower=1.0e8, include_lower=False),
            ),
        ),
        _nonnegative_axis("rank_estimate", units="matrix rank"),
        _nonnegative_axis("nullity_estimate", units="matrix nullity"),
        ParameterAxis(
            "rhs_consistency_defect",
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
    return (
        DerivedParameterRegion(
            "linear_system",
            (
                (
                    ComparisonPredicate("map_linearity_defect", "<=", _LINEARITY_EPS),
                    AffineComparisonPredicate({"dim_x": 1.0, "dim_y": -1.0}, "==", 0.0),
                    MembershipPredicate("residual_target_available", frozenset({True})),
                    MembershipPredicate(
                        "acceptance_relation",
                        frozenset({"residual_below_tolerance"}),
                    ),
                ),
            ),
        ),
        DerivedParameterRegion(
            "least_squares",
            (
                (
                    ComparisonPredicate("map_linearity_defect", "<=", _LINEARITY_EPS),
                    MembershipPredicate(
                        "objective_relation", frozenset({"least_squares"})
                    ),
                    MembershipPredicate("residual_target_available", frozenset({True})),
                ),
            ),
        ),
        DerivedParameterRegion(
            "nonlinear_root",
            (
                (
                    ComparisonPredicate("map_linearity_defect", ">", _LINEARITY_EPS),
                    MembershipPredicate(
                        "acceptance_relation",
                        frozenset({"residual_below_tolerance"}),
                    ),
                ),
                (
                    EvidencePredicate(
                        "map_linearity_defect", frozenset({"unavailable"})
                    ),
                    MembershipPredicate(
                        "acceptance_relation",
                        frozenset({"residual_below_tolerance"}),
                    ),
                ),
                (
                    MembershipPredicate("target_is_zero", frozenset({True})),
                    MembershipPredicate(
                        "acceptance_relation",
                        frozenset({"residual_below_tolerance"}),
                    ),
                ),
            ),
        ),
        DerivedParameterRegion(
            "eigenproblem",
            (
                (
                    ComparisonPredicate("auxiliary_scalar_count", ">=", 1),
                    ComparisonPredicate("normalization_constraint_count", ">=", 1),
                    MembershipPredicate(
                        "acceptance_relation",
                        frozenset({"eigenpair_residual"}),
                    ),
                ),
            ),
        ),
    )


def solve_relation_parameter_schema() -> ParameterSpaceSchema:
    """Return the primitive solve-relation parameter-space schema."""
    return ParameterSpaceSchema(
        name="solve_relation",
        axes=_solve_relation_axes(),
        derived_regions=_solve_relation_regions(),
        invalid_cells=(
            InvalidCellRule(
                "eigenpair_requires_normalization",
                (
                    MembershipPredicate(
                        "acceptance_relation",
                        frozenset({"eigenpair_residual"}),
                    ),
                    ComparisonPredicate("normalization_constraint_count", "==", 0),
                ),
                "eigenpair residuals require a normalization constraint",
            ),
            InvalidCellRule(
                "eigenpair_requires_auxiliary_scalar",
                (
                    MembershipPredicate(
                        "acceptance_relation",
                        frozenset({"eigenpair_residual"}),
                    ),
                    ComparisonPredicate("auxiliary_scalar_count", "==", 0),
                ),
                "eigenpair residuals require a spectral auxiliary scalar",
            ),
        ),
    )


def linear_solver_parameter_schema() -> ParameterSpaceSchema:
    """Return the solve-relation schema extended with linear-operator axes."""
    return ParameterSpaceSchema(
        name="linear_solver",
        axes=_solve_relation_axes() + _linear_operator_axes(),
        derived_regions=_solve_relation_regions()
        + (
            DerivedParameterRegion(
                "square",
                (
                    (
                        AffineComparisonPredicate(
                            {"dim_x": 1.0, "dim_y": -1.0}, "==", 0.0
                        ),
                    ),
                ),
            ),
            DerivedParameterRegion(
                "overdetermined",
                (
                    (
                        AffineComparisonPredicate(
                            {"dim_y": 1.0, "dim_x": -1.0}, ">", 0.0
                        ),
                    ),
                ),
            ),
            DerivedParameterRegion(
                "full_rank",
                ((ComparisonPredicate("singular_value_lower_bound", ">", 0.0),),),
            ),
            DerivedParameterRegion(
                "rank_deficient",
                ((ComparisonPredicate("nullity_estimate", ">", 0),),),
            ),
            DerivedParameterRegion(
                "symmetric_positive_definite",
                (
                    (
                        AffineComparisonPredicate(
                            {"dim_x": 1.0, "dim_y": -1.0}, "==", 0.0
                        ),
                        ComparisonPredicate("symmetry_defect", "<=", _LINEARITY_EPS),
                        ComparisonPredicate("coercivity_lower_bound", ">", 0.0),
                    ),
                ),
            ),
            DerivedParameterRegion(
                "matrix_free",
                (
                    (
                        MembershipPredicate(
                            "linear_operator_matrix_available", frozenset({False})
                        ),
                        MembershipPredicate(
                            "operator_application_available", frozenset({True})
                        ),
                    ),
                ),
            ),
        ),
        invalid_cells=(
            InvalidCellRule(
                "coercivity_requires_square_map",
                (
                    AffineComparisonPredicate({"dim_x": 1.0, "dim_y": -1.0}, "!=", 0.0),
                    ComparisonPredicate("coercivity_lower_bound", ">", 0.0),
                ),
                "positive coercivity is meaningful only for square maps",
            ),
            InvalidCellRule(
                "symmetry_requires_square_map",
                (
                    AffineComparisonPredicate({"dim_x": 1.0, "dim_y": -1.0}, "!=", 0.0),
                    ComparisonPredicate("symmetry_defect", "<=", _LINEARITY_EPS),
                ),
                "matrix symmetry is meaningful only for square maps",
            ),
        ),
    )


def decomposition_parameter_schema() -> ParameterSpaceSchema:
    """Return the dense-matrix decomposition parameter-space schema."""
    return ParameterSpaceSchema(
        name="decomposition",
        axes=(
            _positive_axis("matrix_rows", units="rows"),
            _positive_axis("matrix_columns", units="columns"),
            _positive_axis(
                "factorization_work_budget_fmas", units="fused multiply-adds"
            ),
            _positive_axis("factorization_memory_budget_bytes", units="bytes"),
        )
        + _linear_operator_axes(),
        derived_regions=(
            DerivedParameterRegion(
                "square",
                (
                    (
                        AffineComparisonPredicate(
                            {"matrix_rows": 1.0, "matrix_columns": -1.0},
                            "==",
                            0.0,
                        ),
                    ),
                ),
            ),
            DerivedParameterRegion(
                "full_rank",
                ((ComparisonPredicate("singular_value_lower_bound", ">", 0.0),),),
            ),
            DerivedParameterRegion(
                "rank_deficient",
                ((ComparisonPredicate("nullity_estimate", ">", 0),),),
            ),
        ),
        invalid_cells=(
            InvalidCellRule(
                "coercivity_requires_square_matrix",
                (
                    AffineComparisonPredicate(
                        {"matrix_rows": 1.0, "matrix_columns": -1.0}, "!=", 0.0
                    ),
                    ComparisonPredicate("coercivity_lower_bound", ">", 0.0),
                ),
                "positive coercivity is meaningful only for square matrices",
            ),
        ),
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
        schema="linear_solver",
        coordinates={
            "dim_x": DescriptorCoordinate(n),
            "dim_y": DescriptorCoordinate(n),
            "auxiliary_scalar_count": DescriptorCoordinate(0),
            "equality_constraint_count": DescriptorCoordinate(0),
            "normalization_constraint_count": DescriptorCoordinate(0),
            "residual_target_available": DescriptorCoordinate(True),
            "target_is_zero": DescriptorCoordinate(False),
            "map_linearity_defect": DescriptorCoordinate(0.0),
            "matrix_representation_available": DescriptorCoordinate(True),
            "operator_application_available": DescriptorCoordinate(True),
            "derivative_oracle_kind": DescriptorCoordinate("matrix"),
            "objective_relation": DescriptorCoordinate("none"),
            "acceptance_relation": DescriptorCoordinate("residual_below_tolerance"),
            "requested_residual_tolerance": DescriptorCoordinate(
                requested_residual_tolerance
            ),
            "requested_solution_tolerance": DescriptorCoordinate(
                requested_solution_tolerance
            ),
            "backend_kind": DescriptorCoordinate(_backend_kind(b.backend)),
            "device_kind": DescriptorCoordinate(device_kind),
            "work_budget_fmas": DescriptorCoordinate(work_budget_fmas),
            "memory_budget_bytes": DescriptorCoordinate(memory_budget_bytes),
            "linear_operator_matrix_available": DescriptorCoordinate(True),
            "assembly_cost_fmas": DescriptorCoordinate(assembly_cost_fmas),
            "matvec_cost_fmas": DescriptorCoordinate(matvec_cost_fmas),
            "linear_operator_memory_bytes": DescriptorCoordinate(memory_estimate),
            "symmetry_defect": DescriptorCoordinate(symmetry_defect),
            "skew_symmetry_defect": DescriptorCoordinate(skew_symmetry_defect),
            "diagonal_nonzero_margin": DescriptorCoordinate(diagonal_nonzero_margin),
            "diagonal_dominance_margin": DescriptorCoordinate(
                diagonal_dominance_margin
            ),
            "coercivity_lower_bound": DescriptorCoordinate(
                coercivity_lower_bound, evidence="lower_bound"
            ),
            "singular_value_lower_bound": DescriptorCoordinate(
                singular_value_lower_bound, evidence="lower_bound"
            ),
            "condition_estimate": DescriptorCoordinate(
                condition_estimate, evidence="estimate"
            ),
            "rank_estimate": DescriptorCoordinate(rank_estimate, evidence="estimate"),
            "nullity_estimate": DescriptorCoordinate(
                nullity_estimate, evidence="estimate"
            ),
            "rhs_consistency_defect": DescriptorCoordinate(rhs_consistency_defect),
        },
    )
    return LinearOperatorDescriptor(descriptor, matrix)


__all__ = [
    "AffineComparisonPredicate",
    "AlgorithmCapability",
    "AlgorithmRegistry",
    "AlgorithmRequest",
    "AlgorithmStructureContract",
    "ComparisonPredicate",
    "coverage_patches_are_disjoint",
    "CoveragePatch",
    "DerivedParameterRegion",
    "DescriptorCoordinate",
    "decomposition_parameter_schema",
    "EvidencePredicate",
    "InvalidCellRule",
    "LinearOperatorDescriptor",
    "linear_solver_parameter_schema",
    "linear_operator_descriptor_from_assembled_operator",
    "MembershipPredicate",
    "NumericInterval",
    "ParameterAxis",
    "ParameterBin",
    "ParameterDescriptor",
    "ParameterPredicate",
    "ParameterSpaceSchema",
    "predicate_sets_are_disjoint",
    "SmallLinearOperator",
    "solve_relation_parameter_schema",
]
