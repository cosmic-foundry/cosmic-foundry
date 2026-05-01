"""State-domain predicates for adaptive time integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

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


class StepLimitingDomain(StateDomain, Protocol):
    """State domain that can estimate a conservative step-size limit."""

    def step_limit(
        self,
        u: Tensor,
        du: Tensor,
        *,
        safety: float = 0.9,
    ) -> float | None:
        """Return a positive step limit from ``u`` and direction ``du``."""


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

    def step_limit(
        self,
        u: Tensor,
        du: Tensor,
        *,
        safety: float = 0.9,
    ) -> float | None:
        """Estimate time to the nonnegative boundary along ``du``."""
        if u.shape != (self.n_components,) or du.shape != (self.n_components,):
            return None

        limit: float | None = None
        for i in range(self.n_components):
            slope = float(du[i])
            if slope >= 0.0:
                continue
            distance = float(u[i]) + self.roundoff_tolerance
            candidate = 0.0 if distance <= 0.0 else safety * distance / -slope
            limit = candidate if limit is None else min(limit, candidate)
        return limit


def check_state_domain(rhs: Any, u: Tensor) -> DomainCheck:
    """Return the RHS state-domain result, accepting when no domain is exposed."""
    domain = getattr(rhs, "state_domain", None)
    if domain is None:
        return DomainCheck(accepted=True)
    return cast(StateDomain, domain).check(u)


def predict_domain_step_limit(
    rhs: Any,
    t: float,
    u: Tensor,
    *,
    safety: float = 0.9,
) -> float | None:
    """Return a domain-implied step limit for ``rhs`` at ``(t, u)`` if known."""
    domain = getattr(rhs, "state_domain", None)
    if domain is None or not hasattr(domain, "step_limit"):
        return None
    du = rhs(t, u)
    return cast(StepLimitingDomain, domain).step_limit(u, du, safety=safety)


__all__ = [
    "check_state_domain",
    "DomainCheck",
    "DomainViolation",
    "NonnegativeStateDomain",
    "predict_domain_step_limit",
    "StateDomain",
    "StepLimitingDomain",
]
