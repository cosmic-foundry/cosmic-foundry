"""TimeStepper and IntegratorSelectionResult."""

from __future__ import annotations

from typing import NamedTuple

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.integrator import (
    ConstantStep,
    Controller,
    RHSProtocol,
    RKState,
    TimeIntegrator,
)


class IntegratorSelectionResult(NamedTuple):
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


class TimeStepper:
    """Drives a TimeIntegrator forward in time from t0 to t_end.

    Calls integrator.step() at each time level using the controller's
    suggested step size.  The last step is shortened so that the final
    time is exactly t_end.

    Parameters
    ----------
    integrator:
        A TimeIntegrator instance.
    controller:
        A Controller that suggests the step size.  Defaults to
        ConstantStep(dt) when dt is provided.
    dt:
        Convenience shorthand: if supplied and controller is None, wraps
        dt in a ConstantStep.  Exactly one of controller or dt must be given.
    """

    def __init__(
        self,
        integrator: TimeIntegrator,
        controller: Controller | None = None,
        dt: float | None = None,
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

    def advance(
        self,
        rhs: RHSProtocol,
        u0: Tensor,
        t0: float,
        t_end: float,
    ) -> RKState:
        """Advance from t0 to t_end; return the final RKState.

        Parameters
        ----------
        rhs:
            The ODE right-hand side.
        u0:
            Initial state vector as a Tensor.
        t0:
            Initial time.
        t_end:
            Final time.  The last step is shortened so that t = t_end exactly.
        """
        state = RKState(t0, u0)
        while state.t < t_end:
            dt = self._controller.suggest(state)
            dt = min(dt, t_end - state.t)
            state = self._integrator.step(rhs, state, dt)
        return state


__all__ = ["IntegratorSelectionResult", "TimeStepper"]
