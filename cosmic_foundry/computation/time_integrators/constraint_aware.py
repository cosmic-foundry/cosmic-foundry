"""ConstraintAwareController: lifecycle management for algebraic constraints.

For reaction networks near equilibrium, individual reaction pairs cross the
equilibrium point (r⁺ⱼ = r⁻ⱼ) and can be treated as algebraic constraints
rather than stiff ODEs.  This controller wraps a step-size controller and
manages that lifecycle between accepted steps:

- activates a constraint when |r⁺ⱼ − r⁻ⱼ| / max(r⁺ⱼ, r⁻ⱼ) < ε_activate;
- deactivates when the ratio rises above ε_deactivate (hysteresis prevents
  chattering at the transition);
- applies consistent initialization (linearized projection onto the newly-
  activated constraint manifold) before the next step;
- calls ``project_conserved`` after every accepted step to maintain exact
  conservation to floating-point precision.
"""

from __future__ import annotations

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.implicit import (
    ImplicitRungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    PIController,
)
from cosmic_foundry.computation.time_integrators.reaction_network import (
    ReactionNetworkRHS,
    project_conserved,
)

_RATE_FLOOR = 1e-100  # avoids 0/0 when both rates are zero


def build_constraint_gradients(
    rhs: ReactionNetworkRHS,
    active: frozenset[int],
    t: float,
    u: Tensor,
    eps: float = 1e-7,
) -> Tensor | None:
    """Return the k × n gradient matrix for the active constraint set.

    Row r of the result is ∂(r⁺_j − r⁻_j)/∂u evaluated at (t, u) via
    forward finite differences, where j = sorted(active)[r].  Returns
    ``None`` when ``active`` is empty.

    Parameters
    ----------
    rhs:
        Reaction-network RHS (provides ``forward_rate`` and
        ``reverse_rate`` callables).
    active:
        Frozenset of reaction-pair indices currently treated as constraints.
    t:
        Time at which to evaluate the rate gradients.
    u:
        State vector at which to evaluate the rate gradients.
    eps:
        Finite-difference step size.
    """
    if not active:
        return None
    n = u.shape[0]
    backend = u.backend
    indices = sorted(active)
    k = len(indices)
    r0_plus = rhs.forward_rate(t, u)
    r0_minus = rhs.reverse_rate(t, u)

    # grad[row_idx][k_idx] = d(r⁺_j − r⁻_j)/du_{k_idx}
    grad = [[0.0] * n for _ in range(k)]
    for k_idx in range(n):
        e_k = Tensor.zeros(n, backend=backend)
        e_k = e_k.set(k_idx, Tensor(eps, backend=backend))
        r_plus_p = rhs.forward_rate(t, u + e_k)
        r_minus_p = rhs.reverse_rate(t, u + e_k)
        for row_idx, j in enumerate(indices):
            grad[row_idx][k_idx] = (
                float(r_plus_p[j])
                - float(r_minus_p[j])
                - float(r0_plus[j])
                + float(r0_minus[j])
            ) / eps

    return Tensor(grad, backend=backend)


def _equilibrium_ratios(
    rhs: ReactionNetworkRHS,
    t: float,
    u: Tensor,
    eligible: list[int],
) -> dict[int, float]:
    """Return |r⁺_j − r⁻_j| / max(r⁺_j, r⁻_j) for each j in eligible."""
    r_plus = rhs.forward_rate(t, u)
    r_minus = rhs.reverse_rate(t, u)
    return {
        j: abs(float(r_plus[j]) - float(r_minus[j]))
        / max(abs(float(r_plus[j])), abs(float(r_minus[j])), _RATE_FLOOR)
        for j in eligible
    }


def _consistent_init(
    rhs: ReactionNetworkRHS,
    u: Tensor,
    new_active: frozenset[int],
    t: float,
    eps: float = 1e-7,
) -> Tensor:
    """Project u onto the manifold defined by all active constraints.

    Builds an augmented basis from conservation rows (when present) and one
    gradient row per newly-active constraint j, ∂(r⁺_j − r⁻_j)/∂u.  The
    target for constraint row j is (g_j · u − c_j(u)), which drives
    c_j(u′) ≈ 0 under a first-order linearization.  For linear rate laws
    the projection is exact.
    """
    n_species = u.shape[0]
    backend = u.backend
    cg = build_constraint_gradients(rhs, new_active, t, u, eps)
    if cg is None:
        return u

    indices = sorted(new_active)
    k = len(indices)
    r_plus = rhs.forward_rate(t, u)
    r_minus = rhs.reverse_rate(t, u)

    cg_rows = [[float(cg[r, s]) for s in range(n_species)] for r in range(k)]
    cg_tgt = [
        sum(float(cg[r, s]) * float(u[s]) for s in range(n_species))
        - (float(r_plus[indices[r]]) - float(r_minus[indices[r]]))
        for r in range(k)
    ]

    if rhs.n_conserved > 0:
        n_cons = rhs.n_conserved
        cons_rows = [
            [float(rhs.conservation_basis[i, s]) for s in range(n_species)]
            for i in range(n_cons)
        ]
        cons_tgt = [float(rhs.conservation_targets[i]) for i in range(n_cons)]
        all_rows = cons_rows + cg_rows
        all_tgt = cons_tgt + cg_tgt
    else:
        all_rows = cg_rows
        all_tgt = cg_tgt

    aug_basis: Tensor = Tensor(all_rows, backend=backend)
    aug_targets: Tensor = Tensor(all_tgt, backend=backend)
    return project_conserved(u, aug_basis, aug_targets)


def _update_active_set(
    rhs: ReactionNetworkRHS,
    t: float,
    u: Tensor,
    current: frozenset[int],
    eps_activate: float,
    eps_deactivate: float,
) -> frozenset[int]:
    """Return the updated active set using hysteresis thresholds.

    Eligible pairs are those in ``rhs.constraint_basis`` (pre-screened for
    linear independence).  A pair activates when its equilibrium ratio falls
    below ``eps_activate`` and deactivates when it rises above
    ``eps_deactivate``.  In the hysteresis zone [eps_activate, eps_deactivate]
    the pair remains in its current state.
    """
    eligible = [
        int(float(rhs.constraint_basis[i]))
        for i in range(rhs.constraint_basis.shape[0])
    ]
    if not eligible:
        return frozenset()
    ratios = _equilibrium_ratios(rhs, t, u, eligible)
    new_active: set[int] = set()
    for j in eligible:
        ratio = ratios[j]
        if j in current:
            if ratio <= eps_deactivate:
                new_active.add(j)
        else:
            if ratio < eps_activate:
                new_active.add(j)
    return frozenset(new_active)


class ConstraintAwareController:
    """Wraps a PIController and adds constraint lifecycle management.

    At each accepted step the controller:

    1. Calls ``project_conserved`` to enforce conservation laws exactly.
    2. Evaluates |r⁺ⱼ − r⁻ⱼ| / max(r⁺ⱼ, r⁻ⱼ) for all eligible pairs.
    3. Activates a pair when the ratio falls below ``eps_activate``;
       deactivates when it rises above ``eps_deactivate`` (hysteresis).
    4. On activation, projects the state onto the constraint manifold
       (consistent initialization) before the next step.

    ``advance`` owns the integration loop and is not compatible with
    ``TimeStepper`` because it modifies state between steps.

    Parameters
    ----------
    rhs:
        The reaction-network RHS.
    integrator:
        Implicit Runge-Kutta integrator used for all steps.
    inner:
        Step-size controller.
    eps_activate:
        Equilibrium ratio threshold below which a constraint activates.
        Default 0.01.
    eps_deactivate:
        Equilibrium ratio threshold above which a constraint deactivates.
        Must exceed ``eps_activate`` to provide hysteresis.  Default 0.1.
    eps_grad:
        Finite-difference step for constraint gradient computation.
    """

    def __init__(
        self,
        rhs: ReactionNetworkRHS,
        integrator: ImplicitRungeKuttaIntegrator,
        inner: PIController,
        *,
        eps_activate: float = 0.01,
        eps_deactivate: float = 0.1,
        eps_grad: float = 1e-7,
    ) -> None:
        if eps_deactivate <= eps_activate:
            raise ValueError("eps_deactivate must exceed eps_activate for hysteresis.")
        self._rhs = rhs
        self._integrator = integrator
        self._inner = inner
        self._eps_activate = eps_activate
        self._eps_deactivate = eps_deactivate
        self._eps_grad = eps_grad

        # Diagnostic event log.
        self.activation_events: list[tuple[float, frozenset[int]]] = []
        self.deactivation_events: list[tuple[float, frozenset[int]]] = []

    def advance(
        self,
        u0: Tensor,
        t0: float,
        t_end: float,
        *,
        initial_active: frozenset[int] | None = None,
    ) -> ODEState:
        """Advance from t0 to t_end with constraint lifecycle management.

        Parameters
        ----------
        u0:
            Initial state vector.
        t0:
            Initial time.
        t_end:
            Final time.
        initial_active:
            Initial active constraint set.  Defaults to no active constraints.
        """
        active: frozenset[int] = (
            initial_active if initial_active is not None else frozenset()
        )
        state = ODEState(t0, u0, active_constraints=active)
        dt = self._inner.suggest(state, accepted=True)

        while state.t < t_end:
            dt_try = min(dt, t_end - state.t)
            cg = build_constraint_gradients(
                self._rhs,
                state.active_constraints,  # type: ignore[arg-type]
                state.t,
                state.u,
                self._eps_grad,
            )
            candidate = self._integrator.step(
                self._rhs, state, dt_try, constraint_gradients=cg
            )
            accepted = self._inner.accept(candidate)
            dt = self._inner.suggest(candidate, accepted=accepted)
            if not accepted:
                continue

            # Enforce conservation laws on accepted u.
            u_proj = (
                project_conserved(
                    candidate.u,
                    self._rhs.conservation_basis,
                    self._rhs.conservation_targets,
                )
                if self._rhs.n_conserved > 0
                else candidate.u
            )

            # Evaluate constraint ratios and update active set.
            prev_active: frozenset[int] = state.active_constraints  # type: ignore[assignment]
            new_active = _update_active_set(
                self._rhs,
                candidate.t,
                u_proj,
                prev_active,
                self._eps_activate,
                self._eps_deactivate,
            )

            # Consistent initialization on newly activated constraints.
            newly_on = new_active - prev_active
            newly_off = prev_active - new_active
            if newly_on:
                self.activation_events.append((candidate.t, newly_on))
                u_proj = _consistent_init(
                    self._rhs, u_proj, new_active, candidate.t, self._eps_grad
                )
            if newly_off:
                self.deactivation_events.append((candidate.t, newly_off))

            state = candidate._replace(u=u_proj, active_constraints=new_active)

        return state


__all__ = ["ConstraintAwareController", "build_constraint_gradients"]
