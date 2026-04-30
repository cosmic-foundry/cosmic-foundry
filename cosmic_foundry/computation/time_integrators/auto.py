"""Automatic time-integrator dispatch by RHS structure."""

from __future__ import annotations

from cosmic_foundry.computation.time_integrators.constraint_aware import (
    build_constraint_gradients,
)
from cosmic_foundry.computation.time_integrators.exponential import (
    CoxMatthewsETDRK4Integrator,
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

DEFAULT_EXPLICIT = RungeKuttaIntegrator(4)
DEFAULT_SEMILINEAR = CoxMatthewsETDRK4Integrator(4)
DEFAULT_SYMPLECTIC = SymplecticCompositionIntegrator(2)
DEFAULT_COMPOSITION = CompositionIntegrator(
    [RungeKuttaIntegrator(4), RungeKuttaIntegrator(4)], order=2
)
DEFAULT_SPLIT = AdditiveRungeKuttaIntegrator(2)
DEFAULT_IMPLICIT = ImplicitRungeKuttaIntegrator(2)


class AutoIntegrator(TimeIntegrator):
    """Dispatch to the integrator family implied by the RHS structure.

    The dispatcher is intentionally conservative: it chooses the first
    matching structural protocol in the roadmap order and forwards the step
    to the corresponding specialist integrator.  The class is a convenience
    wrapper; callers that know the algorithm they want should keep using the
    specialist integrators directly.

    Parameters
    ----------
    explicit:
        Fallback explicit Runge-Kutta integrator.
    semilinear:
        Exponential integrator for ``du/dt = Lu + N(t, u)``.
    symplectic:
        Symplectic composition integrator for Hamiltonian RHS objects.
    composition:
        General composition integrator for operator-split RHS objects.
    split:
        Additive Runge-Kutta integrator for explicit/implicit splits.
    implicit:
        Implicit Runge-Kutta integrator for Jacobian-bearing RHS objects.
    """

    def __init__(
        self,
        *,
        explicit: RungeKuttaIntegrator = DEFAULT_EXPLICIT,
        semilinear: CoxMatthewsETDRK4Integrator = DEFAULT_SEMILINEAR,
        symplectic: SymplecticCompositionIntegrator = DEFAULT_SYMPLECTIC,
        composition: CompositionIntegrator = DEFAULT_COMPOSITION,
        split: AdditiveRungeKuttaIntegrator = DEFAULT_SPLIT,
        implicit: ImplicitRungeKuttaIntegrator = DEFAULT_IMPLICIT,
    ) -> None:
        self._explicit = explicit
        self._semilinear = semilinear
        self._symplectic = symplectic
        self._composition = composition
        self._split = split
        self._implicit = implicit

    @property
    def order(self) -> int:
        """Declared order of the explicit fallback branch."""
        return self._explicit.order

    def step(self, rhs: RHSProtocol, state: ODEState, dt: float) -> ODEState:
        """Advance one step with the branch selected by ``rhs`` structure."""
        if isinstance(rhs, SemilinearRHSProtocol):
            return self._semilinear.step(rhs, state, dt)
        if isinstance(rhs, HamiltonianRHSProtocol):
            return self._symplectic.step(rhs, state, dt)
        if isinstance(rhs, CompositeRHSProtocol):
            return self._composition.step(rhs, state, dt)
        if isinstance(rhs, SplitRHSProtocol):
            return self._split.step(rhs, state, dt)
        if isinstance(rhs, ReactionNetworkRHS):
            active = state.active_constraints or frozenset()
            cg = build_constraint_gradients(rhs, active, state.t, state.u)
            return self._implicit.step(rhs, state, dt, constraint_gradients=cg)
        if isinstance(rhs, WithJacobianRHSProtocol):
            return self._implicit.step(rhs, state, dt)
        return self._explicit.step(rhs, state, dt)


__all__ = ["AutoIntegrator"]
