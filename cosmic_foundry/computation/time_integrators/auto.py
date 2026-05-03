"""Automatic time-integrator selection from RHS structure."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

from cosmic_foundry.computation.time_integrators.capabilities import (
    TimeIntegrationCapability,
    TimeIntegrationRequest,
    composite_map_descriptor,
    derivative_oracle_descriptor,
    hamiltonian_map_descriptor,
    select_time_integrator,
    semilinear_map_descriptor,
    split_map_descriptor,
)
from cosmic_foundry.computation.time_integrators.constraint_aware import (
    build_constraint_gradients,
)
from cosmic_foundry.computation.time_integrators.exponential import (
    LawsonRungeKuttaIntegrator,
    SemilinearRHSProtocol,
)
from cosmic_foundry.computation.time_integrators.imex import (
    AdditiveRungeKuttaIntegrator,
    SplitRHSProtocol,
)
from cosmic_foundry.computation.time_integrators.implicit import (
    ImplicitRungeKuttaIntegrator,
    WithJacobianRHSProtocol,
)
from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    RHSProtocol,
    TimeIntegrator,
)
from cosmic_foundry.computation.time_integrators.reaction_network import (
    ReactionNetworkRHS,
)
from cosmic_foundry.computation.time_integrators.runge_kutta import (
    RungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.splitting import (
    CompositeRHSProtocol,
    CompositionIntegrator,
)
from cosmic_foundry.computation.time_integrators.symplectic import (
    HamiltonianRHSProtocol,
    SymplecticCompositionIntegrator,
)

_T = TypeVar("_T")


def _try(cls: Callable[[int], _T], order: int) -> _T | None:
    try:
        return cls(order)
    except ValueError:
        return None


def _try_composition(order: int) -> CompositionIntegrator | None:
    try:
        sub = RungeKuttaIntegrator(1)
        return CompositionIntegrator([sub, sub], order=order)
    except ValueError:
        return None


def _require(branch: _T | None, name: str, order: int) -> _T:
    if branch is None:
        raise ValueError(
            f"AutoIntegrator(order={order}) has no {name} branch at this order"
        )
    return branch


def _rhs_request(rhs: RHSProtocol, order: int) -> TimeIntegrationRequest:
    if isinstance(rhs, SemilinearRHSProtocol):
        return TimeIntegrationRequest(
            requested_properties=frozenset({"one_step", "exponential", "runge_kutta"}),
            order=order,
            descriptor=semilinear_map_descriptor(),
        )
    if isinstance(rhs, HamiltonianRHSProtocol):
        return TimeIntegrationRequest(
            requested_properties=frozenset({"one_step", "symplectic", "composition"}),
            order=order,
            descriptor=hamiltonian_map_descriptor(),
        )
    if isinstance(rhs, CompositeRHSProtocol):
        return TimeIntegrationRequest(
            requested_properties=frozenset(
                {"one_step", "operator_splitting", "composition"}
            ),
            order=order,
            descriptor=composite_map_descriptor(len(rhs.components)),
        )
    if isinstance(rhs, SplitRHSProtocol):
        return TimeIntegrationRequest(
            requested_properties=frozenset({"one_step", "imex", "runge_kutta"}),
            order=order,
            descriptor=split_map_descriptor(),
        )
    if isinstance(rhs, ReactionNetworkRHS):
        return TimeIntegrationRequest(
            requested_properties=frozenset({"one_step", "implicit", "runge_kutta"}),
            order=order,
            descriptor=derivative_oracle_descriptor(),
        )
    if isinstance(rhs, WithJacobianRHSProtocol):
        return TimeIntegrationRequest(
            requested_properties=frozenset({"one_step", "implicit", "runge_kutta"}),
            order=order,
            descriptor=derivative_oracle_descriptor(),
        )
    return TimeIntegrationRequest(
        available_structure=frozenset({"plain_rhs"}),
        requested_properties=frozenset({"one_step", "explicit", "runge_kutta"}),
        order=order,
    )


class AutoIntegrator(TimeIntegrator):
    """Dispatch to the integrator family implied by the RHS structure.

    Converts the RHS protocol into a capability request, selects the declared
    implementation through the package registry, and forwards the step to that
    specialist integrator.  All branches are constructed at the requested
    ``order``; branches whose algorithm family does not support that order are
    left unset and raise ``ValueError`` if selected at step time.

    Parameters
    ----------
    order:
        Convergence order threaded through all branches.  Branches whose
        algorithm family does not support this order (e.g. IMEX only through
        order 4, symplectic only at {1,2,4,6}) are
        unavailable and raise ``ValueError`` on dispatch.
    explicit:
        Override for the explicit RK branch.
    semilinear:
        Override for the semilinear exponential branch.
    symplectic:
        Override for the symplectic Hamiltonian branch.
    composition:
        Override for the operator-splitting branch.
    split:
        Override for the IMEX additive-RK branch.
    implicit:
        Override for the DIRK implicit branch.
    """

    def __init__(
        self,
        order: int,
        *,
        explicit: RungeKuttaIntegrator | None = None,
        semilinear: LawsonRungeKuttaIntegrator | None = None,
        symplectic: SymplecticCompositionIntegrator | None = None,
        composition: CompositionIntegrator | None = None,
        split: AdditiveRungeKuttaIntegrator | None = None,
        implicit: ImplicitRungeKuttaIntegrator | None = None,
    ) -> None:
        self._order = order
        self._explicit: RungeKuttaIntegrator | None = (
            explicit if explicit is not None else _try(RungeKuttaIntegrator, order)
        )
        self._semilinear: LawsonRungeKuttaIntegrator | None = (
            semilinear
            if semilinear is not None
            else _try(LawsonRungeKuttaIntegrator, order)
        )
        self._symplectic: SymplecticCompositionIntegrator | None = (
            symplectic
            if symplectic is not None
            else _try(SymplecticCompositionIntegrator, order)
        )
        self._composition: CompositionIntegrator | None = (
            composition if composition is not None else _try_composition(order)
        )
        self._split: AdditiveRungeKuttaIntegrator | None = (
            split if split is not None else _try(AdditiveRungeKuttaIntegrator, order)
        )
        self._implicit: ImplicitRungeKuttaIntegrator | None = (
            implicit
            if implicit is not None
            else _try(ImplicitRungeKuttaIntegrator, order)
        )
        self._branches: dict[str, Any] = {
            "RungeKuttaIntegrator": self._explicit,
            "LawsonRungeKuttaIntegrator": self._semilinear,
            "SymplecticCompositionIntegrator": self._symplectic,
            "CompositionIntegrator": self._composition,
            "AdditiveRungeKuttaIntegrator": self._split,
            "ImplicitRungeKuttaIntegrator": self._implicit,
        }

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._order

    def step(self, rhs: RHSProtocol, state: ODEState, dt: float) -> ODEState:
        """Advance one step with the branch selected by ``rhs`` structure."""
        selected = self.select(rhs)
        branch = _require(
            self._branches.get(selected.implementation),
            selected.name,
            self._order,
        )
        if isinstance(rhs, ReactionNetworkRHS):
            active = state.active_constraints or frozenset()
            cg = build_constraint_gradients(rhs, active, state.t, state.u)
            return cast(ODEState, branch.step(rhs, state, dt, constraint_gradients=cg))
        return cast(ODEState, branch.step(rhs, state, dt))

    def select(self, rhs: RHSProtocol) -> TimeIntegrationCapability:
        """Return the registered capability selected for ``rhs``."""
        return select_time_integrator(_rhs_request(rhs, self._order))


__all__ = ["AutoIntegrator"]
