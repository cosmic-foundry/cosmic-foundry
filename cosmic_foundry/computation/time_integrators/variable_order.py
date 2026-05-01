"""Variable-order policy for adaptive Nordsieck integration."""

from __future__ import annotations

from typing import NamedTuple

from cosmic_foundry.computation.tensor import norm
from cosmic_foundry.computation.time_integrators.integrator import ODEState
from cosmic_foundry.computation.time_integrators.nordsieck import NordsieckHistory


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

    def decide(self, state: ODEState) -> OrderDecision:
        """Choose acceptance, next order, and next step size for ``state``."""
        nh: NordsieckHistory = state.history
        q = nh.q
        error = self.normalized_error(state)
        h_next = self.next_step_size(nh.h, q, error)
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

    def normalized_error(self, state: ODEState) -> float:
        """Return the dimensionless LTE proxy for the current order."""
        nh: NordsieckHistory = state.history
        q = nh.q
        scale = self.atol + self.rtol * float(norm(state.u))
        raw = float(norm(nh.z[q])) * nh.h / (q + 1)
        return raw / scale

    def smoothness(self, state: ODEState) -> float:
        """Return the highest-slot smoothness ratio used for order changes."""
        nh: NordsieckHistory = state.history
        q = nh.q
        if q == 0:
            return 0.0
        numerator = float(norm(nh.z[q]))
        denominator = float(norm(nh.z[q - 1]))
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


__all__ = [
    "OrderDecision",
    "OrderSelector",
]
