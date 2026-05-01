"""State-domain predicates for adaptive time integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cosmic_foundry.computation.tensor import Tensor


@dataclass(frozen=True)
class DomainViolation:
    """Metadata for one failed state-domain membership check."""

    component: int | None
    value: float
    lower_bound: float
    tolerance: float
    margin: float
    reason: str


@dataclass(frozen=True)
class DomainCheck:
    """Result of testing whether a candidate state belongs to a domain."""

    accepted: bool
    violation: DomainViolation | None = None

    @property
    def rejected(self) -> bool:
        return not self.accepted


class StateDomain(Protocol):
    """Predicate interface for candidate integrator states."""

    def check(self, u: Tensor) -> DomainCheck:
        """Return whether ``u`` is inside the valid state domain."""


@dataclass(frozen=True)
class NonnegativeStateDomain:
    """Domain requiring every component to be nonnegative within tolerance."""

    n_components: int
    roundoff_tolerance: float = 1e-14

    def check(self, u: Tensor) -> DomainCheck:
        if u.shape != (self.n_components,):
            return DomainCheck(
                accepted=False,
                violation=DomainViolation(
                    component=None,
                    value=float("nan"),
                    lower_bound=0.0,
                    tolerance=self.roundoff_tolerance,
                    margin=float("inf"),
                    reason=(
                        f"expected state shape {(self.n_components,)}, got {u.shape}"
                    ),
                ),
            )

        worst_component = 0
        worst_value = float(u[0]) if self.n_components else 0.0
        for i in range(1, self.n_components):
            value = float(u[i])
            if value < worst_value:
                worst_component = i
                worst_value = value

        margin = -self.roundoff_tolerance - worst_value
        if margin > 0.0:
            return DomainCheck(
                accepted=False,
                violation=DomainViolation(
                    component=worst_component,
                    value=worst_value,
                    lower_bound=0.0,
                    tolerance=self.roundoff_tolerance,
                    margin=margin,
                    reason="negative component below roundoff tolerance",
                ),
            )
        return DomainCheck(accepted=True)


__all__ = [
    "DomainCheck",
    "DomainViolation",
    "NonnegativeStateDomain",
    "StateDomain",
]
