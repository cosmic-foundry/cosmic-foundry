"""Variable-order Nordsieck integration within a single multistep family."""

from __future__ import annotations

from typing import NamedTuple

from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators.integrator import RHSProtocol
from cosmic_foundry.computation.time_integrators.nordsieck import (
    AdamsFamily,
    BDFFamily,
    NordsieckIntegrator,
    NordsieckState,
)


class OrderDecision(NamedTuple):
    """Order-selector decision for one attempted Nordsieck step."""

    accepted: bool
    q_next: int
    h_next: float
    error: float


class OrderSelector:
    """Local variable-order policy for Nordsieck multistep states.

    The selector uses two dimensionless diagnostics from the accepted
    Nordsieck vector:

    * an LTE proxy, ``||z[q]|| h / ((q + 1) scale)``, for acceptance and
      step-size updates;
    * a smoothness ratio, ``||z[q]|| / ||z[q-1]||``, for order changes.

    Smooth histories raise order when the LTE proxy is comfortably below
    tolerance.  Sharpening histories lower order when the highest retained
    derivative becomes too large relative to the previous slot.
    """

    def __init__(
        self,
        q_min: int,
        q_max: int,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        safety: float = 0.9,
        factor_min: float = 0.5,
        factor_max: float = 1.2,
        raise_smoothness: float = 0.06,
        lower_smoothness: float = 0.08,
        raise_error: float = 0.8,
    ) -> None:
        if q_min < 1:
            raise ValueError("q_min must be at least 1.")
        if q_max < q_min:
            raise ValueError("q_max must be greater than or equal to q_min.")
        self.q_min = q_min
        self.q_max = q_max
        self.atol = atol
        self.rtol = rtol
        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.raise_smoothness = raise_smoothness
        self.lower_smoothness = lower_smoothness
        self.raise_error = raise_error

    def decide(self, state: NordsieckState) -> OrderDecision:
        """Choose acceptance, next order, and next step size for ``state``."""
        q = state.q
        error = self.normalized_error(state)
        h_next = self.next_step_size(state.h, q, error)
        if error > 1.0:
            return OrderDecision(
                accepted=False,
                q_next=max(self.q_min, q - 1),
                h_next=h_next,
                error=error,
            )

        smoothness = self.smoothness(state)
        q_next = q
        if smoothness > self.lower_smoothness and q > self.q_min:
            q_next = q - 1
        elif (
            smoothness < self.raise_smoothness
            and error < self.raise_error
            and q < self.q_max
        ):
            q_next = q + 1
        return OrderDecision(True, q_next, h_next, error)

    def normalized_error(self, state: NordsieckState) -> float:
        """Return the dimensionless LTE proxy for the current order."""
        q = state.q
        scale = self.atol + self.rtol * float(norm(state.u))
        raw = float(norm(state.z[q])) * state.h / (q + 1)
        return raw / scale

    def smoothness(self, state: NordsieckState) -> float:
        """Return the highest-slot smoothness ratio used for order changes."""
        q = state.q
        if q == 0:
            return 0.0
        numerator = float(norm(state.z[q]))
        denominator = float(norm(state.z[q - 1]))
        if denominator == 0.0:
            return 0.0 if numerator == 0.0 else float("inf")
        return numerator / denominator

    def next_step_size(self, h: float, q: int, error: float) -> float:
        """Return a bounded controller step-size suggestion."""
        if error <= 0.0:
            factor = self.factor_max
        else:
            factor = self.safety * error ** (-1.0 / (q + 1))
        factor = min(self.factor_max, max(self.factor_min, factor))
        return h * factor


class VariableOrderNordsieckIntegrator:
    """Variable-order wrapper around fixed-order Nordsieck integrators.

    The fixed-order `NordsieckIntegrator` remains responsible for the actual
    BDF or Adams corrector.  This wrapper only attempts a step, asks an
    `OrderSelector` whether to accept it, and applies the Phase-9
    `NordsieckState` order/step transformations before the next attempt.
    """

    def __init__(
        self,
        family: AdamsFamily | BDFFamily,
        selector: OrderSelector,
        *,
        q_initial: int | None = None,
        max_rejections: int = 20,
    ) -> None:
        if selector.q_max > family.q_max:
            raise ValueError("selector q_max exceeds family q_max.")
        self._family = family
        self._selector = selector
        self._q = selector.q_min if q_initial is None else q_initial
        if not selector.q_min <= self._q <= selector.q_max:
            raise ValueError("q_initial must lie inside the selector range.")
        self._max_rejections = max_rejections
        self.accepted_orders: list[int] = []
        self.accepted_step_sizes: list[float] = []
        self.accepted_errors: list[float] = []
        self.accepted_times: list[float] = []
        self.rejected_steps = 0

    @property
    def order(self) -> int:
        """Current selected order."""
        return self._q

    @property
    def selector(self) -> OrderSelector:
        """Order-selection policy."""
        return self._selector

    def init_state(
        self,
        rhs: RHSProtocol,
        t0: float,
        u0: Tensor,
        dt: float,
    ) -> NordsieckState:
        """Initialize a Nordsieck state at the current starting order."""
        return NordsieckIntegrator(self._family, self._q).init_state(rhs, t0, u0, dt)

    def step(
        self,
        rhs: RHSProtocol,
        state: NordsieckState,
        dt: float,
    ) -> NordsieckState:
        """Advance by one accepted variable-order step."""
        q = min(self._q, state.q, self._selector.q_max)
        state = state.change_order(q).rescale_step(dt)
        rejections = 0
        while True:
            candidate = NordsieckIntegrator(self._family, q).step(rhs, state, dt)
            decision = self._selector.decide(candidate)
            if decision.accepted:
                self._q = decision.q_next
                self.accepted_orders.append(self._q)
                self.accepted_step_sizes.append(decision.h_next)
                self.accepted_errors.append(decision.error)
                self.accepted_times.append(candidate.t)
                return candidate.change_order(self._q).rescale_step(decision.h_next)

            rejections += 1
            self.rejected_steps += 1
            if rejections > self._max_rejections:
                raise RuntimeError("variable-order step exceeded rejection limit.")
            q = decision.q_next
            dt = decision.h_next
            state = state.change_order(q).rescale_step(dt)

    def advance(
        self,
        rhs: RHSProtocol,
        u0: Tensor,
        t0: float,
        t_end: float,
        dt0: float,
    ) -> NordsieckState:
        """Advance from ``t0`` to ``t_end`` using variable order and step size."""
        state = self.init_state(rhs, t0, u0, dt0)
        while state.t < t_end:
            dt = min(state.h, t_end - state.t)
            state = self.step(rhs, state, dt)
        return state


__all__ = [
    "OrderDecision",
    "OrderSelector",
    "VariableOrderNordsieckIntegrator",
]
