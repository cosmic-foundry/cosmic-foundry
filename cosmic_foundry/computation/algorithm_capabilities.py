"""Shared algorithm capability declarations and selection."""

from __future__ import annotations

from dataclasses import dataclass


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
    supported_orders: frozenset[int] | None = None
    priority: int | None = None

    def supports(self, request: AlgorithmRequest) -> bool:
        """Return whether this declaration inhabits ``request``."""
        if request.order is not None:
            if self.supported_orders is not None:
                if request.order not in self.supported_orders:
                    return False
            elif self.min_order is not None and self.max_order is not None:
                if not self.min_order <= request.order <= self.max_order:
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


__all__ = [
    "AlgorithmCapability",
    "AlgorithmRegistry",
    "AlgorithmRequest",
    "AlgorithmStructureContract",
]
