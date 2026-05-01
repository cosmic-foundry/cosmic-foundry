"""Generic integration driver and selection result."""

from __future__ import annotations

from typing import NamedTuple

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.domains import (
    DomainViolation,
    check_state_domain,
    predict_domain_step_limit,
)
from cosmic_foundry.computation.time_integrators.integrator import (
    ConstantStep,
    Controller,
    ODEState,
    RHSProtocol,
    TimeIntegrator,
)


class IntegrationSelectionResult(NamedTuple):
    """Result of Autotuner integrator selection for a time-integration problem.

    Analogous to SelectionResult for linear solvers: encodes the chosen
    integrator, the recommended constant step size, and the predicted cost
    (total RHS evaluations over the requested t_span at that step size).

    The predicted_cost is in units of RHS evaluations, not wall time.
    Use it to compare configurations against each other.

    Fields
    ------
    integrator:
        The selected TimeIntegrator instance.
    recommended_dt:
        Constant step size that keeps the CFL number below 1 given the
        problem's spectral_radius.  Computed as CFL / spectral_radius where
        CFL is derived from the integrator's stability region.
    predicted_cost:
        Estimated total RHS evaluations: (t_span / recommended_dt) × s,
        where s is the number of stages of the integrator.
    """

    integrator: TimeIntegrator
    recommended_dt: float
    predicted_cost: float


class IntegrationDriver:
    """Drives a TimeIntegrator forward in time from t0 to t_end.

    At each time level the controller suggests a step size, the integrator
    takes the step, and the controller decides whether to accept it.  On
    rejection the step is retried with the controller's suggested smaller
    step size; the time and state are not advanced.  The last accepted step
    is shortened so that the final time is exactly t_end.

    Parameters
    ----------
    integrator:
        A TimeIntegrator instance.
    controller:
        A Controller that suggests the step size and accepts/rejects steps.
        Defaults to ConstantStep(dt) when dt is provided.
    dt:
        Convenience shorthand: if supplied and controller is None, wraps
        dt in a ConstantStep.  Exactly one of controller or dt must be given.
    max_rejections:
        Maximum number of consecutive rejected attempts before advance fails.
    """

    def __init__(
        self,
        integrator: TimeIntegrator,
        controller: Controller | None = None,
        dt: float | None = None,
        max_rejections: int = 20,
    ) -> None:
        if controller is not None and dt is not None:
            raise ValueError("Provide either controller or dt, not both.")
        if controller is None and dt is None:
            raise ValueError("Provide either controller or dt.")
        self._integrator = integrator
        if controller is not None:
            self._controller: Controller = controller
        else:
            self._controller = ConstantStep(dt)  # type: ignore[arg-type]
        self._max_rejections = max_rejections
        self.rejection_reasons: list[str] = []
        self.domain_violations: list[DomainViolation] = []
        self.domain_rejection_step_sizes: list[float] = []
        self.domain_limited_step_sizes: list[float] = []
        self.rejected_steps = 0

    def advance(
        self,
        rhs: RHSProtocol,
        u0: Tensor,
        t0: float,
        t_end: float,
    ) -> ODEState:
        """Advance from t0 to t_end; return the final ODEState.

        Parameters
        ----------
        rhs:
            The ODE right-hand side.
        u0:
            Initial state vector as a Tensor.
        t0:
            Initial time.
        t_end:
            Final time.  The last accepted step is shortened so that
            t = t_end exactly.
        """
        state = ODEState(t0, u0)
        dt = self._controller.suggest(state, accepted=True)
        rejections = 0
        while state.t < t_end:
            dt_try = min(dt, t_end - state.t)
            dt_try = self._limit_step_to_domain(rhs, state, dt_try)
            candidate = self._integrator.step(rhs, state, dt_try)
            error_accepted = self._controller.accept(candidate)
            domain_check = check_state_domain(rhs, candidate.u)
            accepted = error_accepted and domain_check.accepted
            if accepted:
                state = candidate
                rejections = 0
                dt = self._controller.suggest(candidate, accepted=True)
                continue

            rejections += 1
            self.rejected_steps += 1
            if rejections > self._max_rejections:
                raise RuntimeError("integrator step exceeded rejection limit.")
            if not error_accepted:
                self.rejection_reasons.append("error")
                dt = self._controller.suggest(candidate, accepted=False)
                continue

            self.rejection_reasons.append("domain")
            assert domain_check.violation is not None
            self.domain_violations.append(domain_check.violation)
            self.domain_rejection_step_sizes.append(dt_try)
            factor_min = getattr(self._controller, "factor_min", 0.5)
            dt = dt_try * factor_min
        return state

    def _limit_step_to_domain(
        self,
        rhs: RHSProtocol,
        state: ODEState,
        dt: float,
    ) -> float:
        limit = predict_domain_step_limit(rhs, state.t, state.u)
        if limit is None or limit <= 0.0 or limit >= dt:
            return dt
        self.domain_limited_step_sizes.append(limit)
        return limit


__all__ = ["IntegrationSelectionResult", "IntegrationDriver"]
