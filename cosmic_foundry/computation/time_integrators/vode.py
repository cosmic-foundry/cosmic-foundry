"""VODE-style adaptive Nordsieck controller."""

from __future__ import annotations

from typing import Any, cast

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.domains import (
    DomainCheck,
    DomainViolation,
    StateDomain,
)
from cosmic_foundry.computation.time_integrators.implicit import WithJacobianRHSProtocol
from cosmic_foundry.computation.time_integrators.integrator import ODEState
from cosmic_foundry.computation.time_integrators.nordsieck import (
    MultistepIntegrator,
    NordsieckHistory,
)
from cosmic_foundry.computation.time_integrators.stiffness import (
    FamilyName,
    StiffnessDiagnostic,
    StiffnessSwitcher,
)
from cosmic_foundry.computation.time_integrators.variable_order import OrderSelector


class VODEController:
    """Adaptive Adams/BDF controller over Nordsieck states.

    This composes the Phase-10 `OrderSelector` with the Phase-11
    `StiffnessSwitcher`: every attempted step is taken by the currently
    selected Nordsieck family and order; the order selector accepts/rejects
    and proposes the next order/step size; the stiffness switcher then chooses
    whether the next accepted step should use Adams or BDF.
    """

    def __init__(
        self,
        *,
        order_selector: OrderSelector,
        stiffness_switcher: StiffnessSwitcher,
        diagnostic: StiffnessDiagnostic | None = None,
        q_initial: int | None = None,
        initial_family: FamilyName = "adams",
        max_rejections: int = 20,
    ) -> None:
        if order_selector.q_max > 6:
            raise ValueError("order selector q_max exceeds family q_max (6).")
        self._order_selector = order_selector
        self._stiffness_switcher = stiffness_switcher
        self._diagnostic = diagnostic or StiffnessDiagnostic()
        self._q = order_selector.q_min if q_initial is None else q_initial
        if not order_selector.q_min <= self._q <= order_selector.q_max:
            raise ValueError("q_initial must lie inside the selector range.")
        self._family: FamilyName = initial_family
        self._max_rejections = max_rejections

        self.accepted_families: list[FamilyName] = []
        self.accepted_orders: list[int] = []
        self.accepted_step_sizes: list[float] = []
        self.accepted_errors: list[float] = []
        self.accepted_stiffness: list[float] = []
        self.accepted_times: list[float] = []
        self.rejection_reasons: list[str] = []
        self.domain_violations: list[DomainViolation] = []
        self.domain_rejection_step_sizes: list[float] = []
        self.rejected_steps = 0
        self.family_switches = 0

    @property
    def family(self) -> FamilyName:
        """Family selected for the next attempted step."""
        return self._family

    @property
    def order(self) -> int:
        """Order selected for the next attempted step."""
        return self._q

    @property
    def order_selector(self) -> OrderSelector:
        """Local order and step-size selector."""
        return self._order_selector

    @property
    def stiffness_switcher(self) -> StiffnessSwitcher:
        """Family-selection policy."""
        return self._stiffness_switcher

    @property
    def diagnostic(self) -> StiffnessDiagnostic:
        """Streaming stiffness diagnostic."""
        return self._diagnostic

    def init_state(
        self,
        rhs: WithJacobianRHSProtocol,
        t0: float,
        u0: Tensor,
        dt: float,
    ) -> ODEState:
        """Initialize with the current family and order."""
        return self._integrator(self._family, self._q).init_state(rhs, t0, u0, dt)

    def step(
        self,
        rhs: WithJacobianRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        """Advance one accepted adaptive VODE step."""
        nh: NordsieckHistory = state.history
        q = min(self._q, nh.q, self._order_selector.q_max)
        family = self._family
        nh = nh.change_order(q).rescale_step(dt)
        state = ODEState(state.t, state.u, dt, state.err, nh)
        rejections = 0

        while True:
            candidate = self._integrator(family, q).step(rhs, state, dt)
            order_decision = self._order_selector.decide(candidate)
            domain_check = self._domain_check(rhs, candidate.u)
            if order_decision.accepted and domain_check.accepted:
                stiffness = self._diagnostic.update(
                    rhs.jacobian(candidate.t, candidate.u),
                    dt,
                )
                family_decision = self._stiffness_switcher.decide(family, stiffness)
                if family_decision.switched:
                    self.family_switches += 1
                self._family = family_decision.family
                self._q = order_decision.q_next
                self.accepted_families.append(self._family)
                self.accepted_orders.append(self._q)
                self.accepted_step_sizes.append(order_decision.h_next)
                self.accepted_errors.append(order_decision.error)
                self.accepted_stiffness.append(stiffness)
                self.accepted_times.append(candidate.t)
                nh_out = candidate.history.change_order(self._q).rescale_step(
                    order_decision.h_next
                )
                return ODEState(
                    candidate.t,
                    candidate.u,
                    order_decision.h_next,
                    candidate.err,
                    nh_out,
                )

            rejections += 1
            self.rejected_steps += 1
            if not order_decision.accepted:
                self.rejection_reasons.append("error")
            else:
                self.rejection_reasons.append("domain")
                assert domain_check.violation is not None
                self.domain_violations.append(domain_check.violation)
                self.domain_rejection_step_sizes.append(dt)
            if rejections > self._max_rejections:
                raise RuntimeError("VODE step exceeded rejection limit.")
            q = (
                order_decision.q_next
                if not order_decision.accepted
                else max(self._order_selector.q_min, q - 1)
            )
            dt = (
                order_decision.h_next
                if not order_decision.accepted
                else dt * self._order_selector.factor_min
            )
            nh = state.history.change_order(q).rescale_step(dt)
            state = ODEState(state.t, state.u, dt, state.err, nh)

    def advance(
        self,
        rhs: WithJacobianRHSProtocol,
        u0: Tensor,
        t0: float,
        t_end: float,
        dt0: float,
    ) -> ODEState:
        """Advance from ``t0`` to ``t_end`` with adaptive family/order/step."""
        state = self.init_state(rhs, t0, u0, dt0)
        while state.t < t_end:
            dt = min(state.dt, t_end - state.t)
            state = self.step(rhs, state, dt)
        return state

    def _integrator(self, family: FamilyName, q: int) -> MultistepIntegrator:
        return MultistepIntegrator(family, q)

    def _domain_check(self, rhs: Any, u: Tensor) -> DomainCheck:
        domain = getattr(rhs, "state_domain", None)
        if domain is None:
            return DomainCheck(accepted=True)
        return cast(StateDomain, domain).check(u)


__all__ = ["VODEController"]
