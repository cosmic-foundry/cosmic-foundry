"""Stiffness diagnostics and Adams/BDF family switching for Nordsieck states."""

from __future__ import annotations

from typing import Literal, NamedTuple

from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators.implicit import WithJacobianRHSProtocol
from cosmic_foundry.computation.time_integrators.integrator import ODEState
from cosmic_foundry.computation.time_integrators.nordsieck import (
    MultistepIntegrator,
    NordsieckHistory,
)

FamilyName = Literal["adams", "bdf"]


class FamilySwitch(NamedTuple):
    """Family-switch decision for one accepted Nordsieck state."""

    family: FamilyName
    stiffness: float
    switched: bool


class StiffnessDiagnostic:
    """Streaming Gershgorin stiffness estimate from ``hJ``.

    For a dense Jacobian ``J``, the Gershgorin row-sum bound
    ``max_i Σ_j |h J_ij|`` upper-bounds the spectral radius of ``hJ``.  This
    diagnostic stores the latest bound so the switcher can apply hysteresis
    without owning the RHS or the integrator state.
    """

    def __init__(self) -> None:
        self.last: float = 0.0

    def update(self, jacobian: Tensor, h: float) -> float:
        """Update and return the Gershgorin bound for ``h * jacobian``."""
        n_rows = jacobian.shape[0]
        n_cols = jacobian.shape[1]
        max_row = 0.0
        for i in range(n_rows):
            row_sum = 0.0
            for j in range(n_cols):
                row_sum += abs(float(jacobian[i, j]))
            max_row = max(max_row, abs(h) * row_sum)
        self.last = max_row
        return max_row


class StiffnessSwitcher:
    """Hysteresis policy for Adams/BDF family selection.

    The policy switches Adams to BDF when the latest stiffness estimate exceeds
    ``stiff_threshold`` and switches BDF back to Adams only after it falls below
    ``nonstiff_threshold``.  The gap prevents method chatter near the boundary.
    """

    def __init__(
        self,
        *,
        stiff_threshold: float = 1.0,
        nonstiff_threshold: float = 0.5,
    ) -> None:
        if nonstiff_threshold >= stiff_threshold:
            raise ValueError("nonstiff_threshold must be below stiff_threshold.")
        self.stiff_threshold = stiff_threshold
        self.nonstiff_threshold = nonstiff_threshold

    def decide(self, current: FamilyName, stiffness: float) -> FamilySwitch:
        """Return the selected family for the latest stiffness estimate."""
        if current == "adams" and stiffness > self.stiff_threshold:
            return FamilySwitch("bdf", stiffness, True)
        if current == "bdf" and stiffness < self.nonstiff_threshold:
            return FamilySwitch("adams", stiffness, True)
        return FamilySwitch(current, stiffness, False)


class FamilySwitchingNordsieckIntegrator:
    """Nordsieck integrator that switches between Adams and BDF families.

    Phase 11 deliberately stops short of a full VODE-style controller.  The
    current order and step size are fixed by the caller; this wrapper only
    monitors the accepted Jacobian stiffness and changes the corrector family
    when the hysteresis policy says to do so.

    The stored Nordsieck vector is family-neutral: ``z[j] = h^j y^(j) / j!``.
    Switching families therefore preserves the vector directly, with ``z[0]``
    unchanged exactly.  The corrector coefficients live in the fixed-order
    `MultistepIntegrator` selected for the next step.
    """

    def __init__(
        self,
        *,
        switcher: StiffnessSwitcher,
        diagnostic: StiffnessDiagnostic | None = None,
        q: int = 2,
        initial_family: FamilyName = "adams",
    ) -> None:
        if q < 1:
            raise ValueError("q must be at least 1.")
        if q > 6:
            raise ValueError("q must be supported by both families.")
        self._switcher = switcher
        self._diagnostic = diagnostic or StiffnessDiagnostic()
        self._q = q
        self._family: FamilyName = initial_family
        self.accepted_families: list[FamilyName] = []
        self.accepted_stiffness: list[float] = []
        self.accepted_times: list[float] = []
        self.switch_count = 0

    @property
    def family(self) -> FamilyName:
        """Family selected for the next step."""
        return self._family

    @property
    def order(self) -> int:
        """Fixed order used by the Phase-11 switcher."""
        return self._q

    @property
    def diagnostic(self) -> StiffnessDiagnostic:
        """Streaming stiffness diagnostic."""
        return self._diagnostic

    def transform_family(
        self,
        state: ODEState,
        target: FamilyName,
    ) -> ODEState:
        """Return ``state`` with its Nordsieck history at the wrapper's fixed order.

        The Nordsieck representation is already scaled-derivative based, so the
        transformation is identity on the history vector except for enforcing
        the wrapper's fixed order.
        """
        if target not in ("adams", "bdf"):
            raise ValueError(f"unknown Nordsieck family {target!r}.")
        nh: NordsieckHistory = state.history.change_order(self._q)
        return ODEState(state.t, state.u, state.dt, state.err, nh)

    def init_state(
        self,
        rhs: WithJacobianRHSProtocol,
        t0: float,
        u0: Tensor,
        dt: float,
    ) -> ODEState:
        """Initialize using the currently selected family."""
        return self._integrator().init_state(rhs, t0, u0, dt)

    def step(
        self,
        rhs: WithJacobianRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        """Advance one step, update stiffness, and switch family if needed."""
        state = self.transform_family(state, self._family)
        nh: NordsieckHistory = state.history.rescale_step(dt)
        state = ODEState(state.t, state.u, dt, state.err, nh)
        candidate = self._integrator().step(rhs, state, dt)
        stiffness = self._diagnostic.update(rhs.jacobian(candidate.t, candidate.u), dt)
        decision = self._switcher.decide(self._family, stiffness)
        if decision.switched:
            self.switch_count += 1
            candidate = self.transform_family(candidate, decision.family)
        self._family = decision.family
        self.accepted_families.append(self._family)
        self.accepted_stiffness.append(stiffness)
        self.accepted_times.append(candidate.t)
        return candidate

    def advance(
        self,
        rhs: WithJacobianRHSProtocol,
        u0: Tensor,
        t0: float,
        t_end: float,
        dt: float,
    ) -> ODEState:
        """Advance from ``t0`` to ``t_end`` with fixed order and step size."""
        state = self.init_state(rhs, t0, u0, dt)
        while state.t < t_end:
            dt_step = min(dt, t_end - state.t)
            state = self.step(rhs, state, dt_step)
        return state

    def _integrator(self) -> MultistepIntegrator:
        return MultistepIntegrator(self._family, self._q)


def nordsieck_solution_distance(lhs: ODEState, rhs: ODEState) -> float:
    """Return ``||lhs.u - rhs.u||`` for family-transform checks."""
    return float(norm(lhs.u - rhs.u))


__all__ = [
    "FamilyName",
    "FamilySwitch",
    "FamilySwitchingNordsieckIntegrator",
    "StiffnessDiagnostic",
    "StiffnessSwitcher",
    "nordsieck_solution_distance",
]
