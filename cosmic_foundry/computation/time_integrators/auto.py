"""Automatic time-integrator selection from RHS structure."""

from __future__ import annotations

from typing import Any, cast

from cosmic_foundry.computation.algorithm_capabilities import AlgorithmRequest
from cosmic_foundry.computation.time_integrators.capabilities import (
    TimeIntegrationCapability,
    composite_map_descriptor_from_rhs,
    derivative_oracle_descriptor,
    hamiltonian_map_descriptor,
    rhs_evaluation_descriptor,
    select_time_integrator,
    semilinear_map_descriptor,
    split_map_descriptor,
)
from cosmic_foundry.computation.time_integrators.exponential import (
    SemilinearRHSProtocol,
)
from cosmic_foundry.computation.time_integrators.imex import SplitRHSProtocol
from cosmic_foundry.computation.time_integrators.implicit import (
    ConstrainedNewtonRHSProtocol,
    WithJacobianRHSProtocol,
)
from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    RHSProtocol,
    TimeIntegrator,
)
from cosmic_foundry.computation.time_integrators.splitting import (
    CompositeRHSProtocol,
)
from cosmic_foundry.computation.time_integrators.symplectic import (
    HamiltonianRHSProtocol,
)


def _rhs_request(rhs: RHSProtocol, order: int) -> AlgorithmRequest:
    if isinstance(rhs, SemilinearRHSProtocol):
        return AlgorithmRequest(
            requested_properties=frozenset({"one_step"}),
            order=order,
            descriptor=semilinear_map_descriptor(),
        )
    if isinstance(rhs, HamiltonianRHSProtocol):
        return AlgorithmRequest(
            requested_properties=frozenset({"one_step"}),
            order=order,
            descriptor=hamiltonian_map_descriptor(),
        )
    if isinstance(rhs, CompositeRHSProtocol):
        return AlgorithmRequest(
            requested_properties=frozenset({"one_step"}),
            order=order,
            descriptor=composite_map_descriptor_from_rhs(rhs),
        )
    if isinstance(rhs, SplitRHSProtocol):
        return AlgorithmRequest(
            requested_properties=frozenset({"one_step"}),
            order=order,
            descriptor=split_map_descriptor(),
        )
    if isinstance(rhs, WithJacobianRHSProtocol):
        return AlgorithmRequest(
            requested_properties=frozenset({"one_step"}),
            order=order,
            descriptor=derivative_oracle_descriptor(),
        )
    return AlgorithmRequest(
        requested_properties=frozenset({"one_step"}),
        order=order,
        descriptor=rhs_evaluation_descriptor(),
    )


class AutoIntegrator(TimeIntegrator):
    """Dispatch to the integrator family implied by the RHS structure.

    Converts the RHS protocol into a capability request, selects the declared
    implementation through the package registry, constructs it from the selected
    capability, and forwards the step to that specialist integrator.

    Parameters
    ----------
    order:
        Convergence order threaded through all branches.  Branches whose
        algorithm family does not support this order (e.g. IMEX only through
        order 4, symplectic only at {1,2,4,6}) are
        unavailable and raise ``ValueError`` on dispatch.
    """

    def __init__(self, order: int) -> None:
        self._order = order

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._order

    def step(self, rhs: RHSProtocol, state: ODEState, dt: float) -> ODEState:
        """Advance one step with the branch selected by ``rhs`` structure."""
        request = _rhs_request(rhs, self._order)
        selected = select_time_integrator(request)
        branch = cast(Any, selected.construct(request))
        if isinstance(rhs, ConstrainedNewtonRHSProtocol):
            active = state.active_constraints or frozenset()
            cg = rhs.constraint_gradients(active, state.t, state.u)
            return cast(
                ODEState,
                branch.step(rhs, state, dt, constraint_gradients=cg),
            )
        return cast(ODEState, branch.step(rhs, state, dt))

    def select(self, rhs: RHSProtocol) -> TimeIntegrationCapability:
        """Return the registered capability selected for ``rhs``."""
        return select_time_integrator(_rhs_request(rhs, self._order))


__all__ = ["AutoIntegrator"]
