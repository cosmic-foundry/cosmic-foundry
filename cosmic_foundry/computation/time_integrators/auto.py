"""Automatic time-integrator dispatch by RHS structure."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

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


class AutoIntegrator(TimeIntegrator):
    """Dispatch to the integrator family implied by the RHS structure.

    Selects the first matching structural protocol in dispatch order and
    forwards the step to the corresponding specialist integrator.  All
    branches are constructed at the requested ``order``; branches whose
    algorithm family does not support that order are left unset and raise
    ``ValueError`` if dispatched to at step time.

    Parameters
    ----------
    order:
        Convergence order threaded through all branches.  Branches whose
        algorithm family does not support this order (e.g. IMEX only at
        order 2, symplectic only at {1,2,4,6,8}) are
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

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._order

    def step(self, rhs: RHSProtocol, state: ODEState, dt: float) -> ODEState:
        """Advance one step with the branch selected by ``rhs`` structure."""
        if isinstance(rhs, SemilinearRHSProtocol):
            branch = self._semilinear
            if branch is None:
                raise ValueError(
                    f"AutoIntegrator(order={self._order}) has no semilinear "
                    "branch at this order"
                )
            return branch.step(rhs, state, dt)
        if isinstance(rhs, HamiltonianRHSProtocol):
            return _require(self._symplectic, "symplectic", self._order).step(
                rhs, state, dt
            )
        if isinstance(rhs, CompositeRHSProtocol):
            return _require(self._composition, "composition", self._order).step(
                rhs, state, dt
            )
        if isinstance(rhs, SplitRHSProtocol):
            return _require(self._split, "split", self._order).step(rhs, state, dt)
        if isinstance(rhs, ReactionNetworkRHS):
            active = state.active_constraints or frozenset()
            cg = build_constraint_gradients(rhs, active, state.t, state.u)
            return _require(self._implicit, "implicit", self._order).step(
                rhs, state, dt, constraint_gradients=cg
            )
        if isinstance(rhs, WithJacobianRHSProtocol):
            return _require(self._implicit, "implicit", self._order).step(
                rhs, state, dt
            )
        return _require(self._explicit, "explicit", self._order).step(rhs, state, dt)


__all__ = ["AutoIntegrator"]
