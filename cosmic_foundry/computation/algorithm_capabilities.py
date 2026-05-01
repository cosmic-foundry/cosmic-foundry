"""Shared algorithm capability declarations and selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias


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
CoverageStatus: TypeAlias = Literal["owned", "rejected"]
CellStatus: TypeAlias = Literal["invalid", "owned", "rejected", "uncovered"]


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


StructuredPredicate: TypeAlias = (
    ComparisonPredicate | MembershipPredicate | AffineComparisonPredicate
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
    """Claimed owned or intentionally rejected parameter-space region."""

    name: str
    owner: str
    status: CoverageStatus
    predicates: tuple[StructuredPredicate, ...]
    priority: int | None = None

    @property
    def referenced_fields(self) -> frozenset[str]:
        """Descriptor fields referenced by this coverage patch."""
        return frozenset().union(
            *(predicate.referenced_fields for predicate in self.predicates)
        )

    def contains(self, descriptor: ParameterDescriptor) -> bool:
        """Return whether ``descriptor`` lies inside this patch."""
        return all(predicate.evaluate(descriptor) for predicate in self.predicates)


@dataclass(frozen=True)
class ParameterSpaceSchema:
    """Parameter-space axes, validity rules, and coverage validation."""

    name: str
    axes: tuple[ParameterAxis, ...]
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

    def cell_status(
        self,
        descriptor: ParameterDescriptor,
        patches: tuple[CoveragePatch, ...],
    ) -> CellStatus:
        """Classify ``descriptor`` as invalid, owned, rejected, or uncovered."""
        self.validate_descriptor(descriptor)
        for patch in patches:
            self.validate_coverage_patch(patch)
        if any(rule.matches(descriptor) for rule in self.invalid_cells):
            return "invalid"
        containing = tuple(patch for patch in patches if patch.contains(descriptor))
        if not containing:
            return "uncovered"
        if all(patch.status == "rejected" for patch in containing):
            return "rejected"
        return "owned"

    @staticmethod
    def _validate_predicate(predicate: StructuredPredicate) -> None:
        allowed = (
            ComparisonPredicate,
            MembershipPredicate,
            AffineComparisonPredicate,
        )
        if not isinstance(predicate, allowed):
            raise TypeError(f"unsupported parameter-space predicate {predicate!r}")


__all__ = [
    "AffineComparisonPredicate",
    "AlgorithmCapability",
    "AlgorithmRegistry",
    "AlgorithmRequest",
    "AlgorithmStructureContract",
    "ComparisonPredicate",
    "CoveragePatch",
    "DescriptorCoordinate",
    "InvalidCellRule",
    "MembershipPredicate",
    "NumericInterval",
    "ParameterAxis",
    "ParameterBin",
    "ParameterDescriptor",
    "ParameterPredicate",
    "ParameterSpaceSchema",
]
