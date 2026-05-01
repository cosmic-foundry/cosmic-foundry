"""Algorithm structure contracts for time-integration selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmStructureContract:
    """Required input structure and provided algorithmic properties."""

    requires: frozenset[str]
    provides: frozenset[str]


@dataclass(frozen=True)
class TimeIntegrationCapability:
    """Declared capability of one selectable time-integration implementation."""

    name: str
    implementation: str
    category: str
    contract: AlgorithmStructureContract
    min_order: int
    max_order: int
    supported_orders: frozenset[int] | None = None
    priority: int | None = None

    def supports(self, request: TimeIntegrationRequest) -> bool:
        """Return whether this declaration inhabits ``request``."""
        if request.order is not None:
            if self.supported_orders is not None:
                if request.order not in self.supported_orders:
                    return False
            elif not self.min_order <= request.order <= self.max_order:
                return False
        return (
            self.contract.requires <= request.available_structure
            and request.requested_properties <= self.contract.provides
        )


@dataclass(frozen=True)
class TimeIntegrationRequest:
    """Requested input structure and desired time-integration properties."""

    available_structure: frozenset[str] = frozenset()
    requested_properties: frozenset[str] = frozenset()
    order: int | None = None


class TimeIntegrationRegistry:
    """Select time-integration implementations by declared capabilities."""

    def __init__(self, capabilities: tuple[TimeIntegrationCapability, ...]) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> tuple[TimeIntegrationCapability, ...]:
        """Registered implementation declarations."""
        return self._capabilities

    def matching(
        self, request: TimeIntegrationRequest
    ) -> tuple[TimeIntegrationCapability, ...]:
        """Return all declarations that inhabit ``request``."""
        return tuple(cap for cap in self._capabilities if cap.supports(request))

    def select(self, request: TimeIntegrationRequest) -> TimeIntegrationCapability:
        """Return the unique or explicitly prioritized implementation."""
        matches = self.matching(request)
        if not matches:
            raise ValueError(f"no time integrator satisfies request {request!r}")
        if len(matches) == 1:
            return matches[0]

        ranked = [cap for cap in matches if cap.priority is not None]
        if not ranked:
            names = ", ".join(cap.name for cap in matches)
            raise ValueError(f"ambiguous time-integrator request {request!r}: {names}")
        ranked.sort(key=lambda cap: cap.priority if cap.priority is not None else 0)
        if len(ranked) > 1 and ranked[0].priority == ranked[1].priority:
            names = ", ".join(cap.name for cap in ranked)
            raise ValueError(f"ambiguous time-integrator priority {request!r}: {names}")
        return ranked[0]


def _contract(
    *,
    requires: tuple[str, ...],
    provides: tuple[str, ...],
) -> AlgorithmStructureContract:
    return AlgorithmStructureContract(frozenset(requires), frozenset(provides))


_CAPABILITIES: tuple[TimeIntegrationCapability, ...] = (
    TimeIntegrationCapability(
        "explicit_runge_kutta",
        "RungeKuttaIntegrator",
        "method_family",
        _contract(
            requires=("plain_rhs",),
            provides=("one_step", "explicit", "runge_kutta"),
        ),
        1,
        6,
        priority=60,
    ),
    TimeIntegrationCapability(
        "implicit_runge_kutta",
        "ImplicitRungeKuttaIntegrator",
        "method_family",
        _contract(
            requires=("jacobian_rhs",),
            provides=("one_step", "implicit", "runge_kutta"),
        ),
        1,
        6,
    ),
    TimeIntegrationCapability(
        "additive_runge_kutta",
        "AdditiveRungeKuttaIntegrator",
        "method_family",
        _contract(
            requires=("split_rhs",),
            provides=("one_step", "imex", "runge_kutta"),
        ),
        1,
        4,
    ),
    TimeIntegrationCapability(
        "lawson_runge_kutta",
        "LawsonRungeKuttaIntegrator",
        "method_family",
        _contract(
            requires=("semilinear_rhs",),
            provides=("one_step", "exponential", "runge_kutta"),
        ),
        1,
        6,
    ),
    TimeIntegrationCapability(
        "symplectic_composition",
        "SymplecticCompositionIntegrator",
        "method_family",
        _contract(
            requires=("hamiltonian_rhs",),
            provides=("one_step", "symplectic", "composition"),
        ),
        1,
        6,
        supported_orders=frozenset({1, 2, 4, 6}),
    ),
    TimeIntegrationCapability(
        "operator_composition",
        "CompositionIntegrator",
        "method_family",
        _contract(
            requires=("composite_rhs",),
            provides=("one_step", "operator_splitting", "composition"),
        ),
        1,
        6,
        supported_orders=frozenset({1, 2, 4, 6}),
    ),
    TimeIntegrationCapability(
        "explicit_multistep",
        "ExplicitMultistepIntegrator",
        "method_family",
        _contract(
            requires=("plain_rhs",),
            provides=("one_step", "explicit", "multistep"),
        ),
        1,
        6,
        priority=50,
    ),
    TimeIntegrationCapability(
        "fixed_order_nordsieck",
        "MultistepIntegrator",
        "method_family",
        _contract(
            requires=("plain_rhs",),
            provides=("one_step", "nordsieck", "fixed_order"),
        ),
        1,
        6,
    ),
    TimeIntegrationCapability(
        "adaptive_nordsieck",
        "AdaptiveNordsieckController",
        "controller",
        _contract(
            requires=("jacobian_rhs", "state_domain"),
            provides=(
                "advance",
                "nordsieck",
                "adaptive_timestep",
                "variable_order",
                "stiffness_switching",
                "domain_aware_acceptance",
            ),
        ),
        1,
        6,
    ),
    TimeIntegrationCapability(
        "generic_integration_driver",
        "IntegrationDriver",
        "driver",
        _contract(
            requires=("plain_rhs", "time_integrator", "controller"),
            provides=("advance", "adaptive_timestep", "domain_aware_acceptance"),
        ),
        1,
        6,
    ),
    TimeIntegrationCapability(
        "constraint_aware_controller",
        "ConstraintAwareController",
        "controller",
        _contract(
            requires=("reaction_network_rhs", "conservation_constraints"),
            provides=("advance", "constraint_lifecycle", "domain_aware_acceptance"),
        ),
        1,
        6,
    ),
)


TIME_INTEGRATION_REGISTRY = TimeIntegrationRegistry(_CAPABILITIES)


def time_integration_capabilities() -> tuple[TimeIntegrationCapability, ...]:
    """Return declared time-integration algorithm capabilities."""
    return TIME_INTEGRATION_REGISTRY.capabilities


def select_time_integrator(
    request: TimeIntegrationRequest,
) -> TimeIntegrationCapability:
    """Select a time-integration implementation declaration by capability."""
    return TIME_INTEGRATION_REGISTRY.select(request)


__all__ = [
    "AlgorithmStructureContract",
    "select_time_integrator",
    "TimeIntegrationCapability",
    "time_integration_capabilities",
    "TimeIntegrationRegistry",
    "TimeIntegrationRequest",
    "TIME_INTEGRATION_REGISTRY",
]
