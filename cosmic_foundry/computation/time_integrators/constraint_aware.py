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

from cosmic_foundry.computation.algorithm_capabilities import (
    AffineComparisonPredicate,
    ComparisonPredicate,
    CoverageRegion,
    ReactionNetworkField,
)
from cosmic_foundry.computation.solvers._root_execution import solve_root_relation
from cosmic_foundry.computation.solvers.newton_root_solver import RootRelation
from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.domains import (
    DomainViolation,
    check_state_domain,
    predict_domain_step_limit,
)
from cosmic_foundry.computation.time_integrators.implicit import (
    ConstrainedNewtonRHSProtocol,
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
_RATE_THRESHOLD = 1e-10  # pair is considered absent; ratio = inf → no activation


def build_constraint_gradients(
    rhs: ConstrainedNewtonRHSProtocol,
    active: frozenset[int],
    t: float,
    u: Tensor,
    eps: float = 1e-7,
) -> Tensor | None:
    """Return gradient rows for active algebraic constraints."""
    return rhs.constraint_gradients(active, t, u, eps)


def _equilibrium_ratios(
    rhs: ReactionNetworkRHS,
    t: float,
    u: Tensor,
    eligible: list[int],
) -> dict[int, float]:
    """Return |r⁺_j − r⁻_j| / max(r⁺_j, r⁻_j) for each j in eligible.

    Returns ``inf`` for pairs where both rates are below ``_RATE_THRESHOLD``
    (absent species), preventing spurious activation of empty-channel pairs.
    """
    r_plus = rhs.forward_rate(t, u)
    r_minus = rhs.reverse_rate(t, u)
    result: dict[int, float] = {}
    for j in eligible:
        rp = abs(float(r_plus[j]))
        rm = abs(float(r_minus[j]))
        denom = max(rp, rm, _RATE_FLOOR)
        if denom < _RATE_THRESHOLD:
            result[j] = float("inf")
        else:
            result[j] = abs(rp - rm) / denom
    return result


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


def nse_root_relation(
    rhs: ReactionNetworkRHS,
    u: Tensor,
    t: float = 0.0,
    *,
    eps: float = 1e-7,
) -> RootRelation:
    """Return the root relation defining the Nuclear Statistical Equilibrium state.

    At NSE every independent reaction pair satisfies r⁺_j = r⁻_j.  The
    equilibrium state is determined by the combined system:

    - Conservation: conservation_basis @ u = conservation_targets
    - Equilibrium: r⁺_j(u) − r⁻_j(u) = 0 for all j in constraint_basis

    When all independent constraints are active, this is a square
    n_species × n_species system; the supplied ``u`` is the Newton initial
    guess.

    Parameters
    ----------
    rhs:
        Reaction-network RHS; provides conservation basis and rate callables.
    u:
        Initial guess for the NSE state, shape (n_species,).
    t:
        Time at which rates are evaluated (for time-dependent rate laws).
    eps:
        Finite-difference step size for Jacobian assembly.

    """
    n_species = u.shape[0]
    backend = u.backend
    active = frozenset(
        int(float(rhs.constraint_basis[i]))
        for i in range(rhs.constraint_basis.shape[0])
    )
    indices = sorted(active)
    n_cons = rhs.n_conserved

    def F(x: Tensor) -> Tensor:
        rows: list[float] = []
        if n_cons > 0:
            for i in range(n_cons):
                val = sum(
                    float(rhs.conservation_basis[i, s]) * float(x[s])
                    for s in range(n_species)
                )
                rows.append(val - float(rhs.conservation_targets[i]))
        r_plus = rhs.forward_rate(t, x)
        r_minus = rhs.reverse_rate(t, x)
        for j in indices:
            rows.append(float(r_plus[j]) - float(r_minus[j]))
        return Tensor(rows, backend=backend)

    def J(x: Tensor) -> Tensor:
        rows: list[list[float]] = []
        if n_cons > 0:
            for i in range(n_cons):
                rows.append(
                    [float(rhs.conservation_basis[i, s]) for s in range(n_species)]
                )
        cg = build_constraint_gradients(rhs, active, t, x, eps)
        if cg is not None:
            k = len(indices)
            for r in range(k):
                rows.append([float(cg[r, s]) for s in range(n_species)])
        return Tensor(rows, backend=backend)

    return RootRelation(F, J, u)


def solve_nse(
    rhs: ReactionNetworkRHS,
    u: Tensor,
    t: float = 0.0,
    *,
    eps: float = 1e-7,
) -> Tensor:
    """Solve for the Nuclear Statistical Equilibrium state."""
    return solve_root_relation(nse_root_relation(rhs, u, t, eps=eps))


class NuclearStatisticalEquilibriumSolver:
    """Solve fully constrained reaction-network equilibrium descriptors."""

    def solve(
        self,
        rhs: ReactionNetworkRHS,
        u: Tensor,
        t: float = 0.0,
        *,
        eps: float = 1e-7,
    ) -> Tensor:
        """Return the NSE state for ``rhs`` from initial guess ``u``."""
        return solve_nse(rhs, u, t, eps=eps)


def reaction_network_coverage_regions() -> tuple[CoverageRegion, ...]:
    """Return descriptor-space regions owned by reaction-network solvers."""
    field = ReactionNetworkField
    return (
        CoverageRegion(
            NuclearStatisticalEquilibriumSolver,
            (
                ComparisonPredicate(field.CONSERVATION_LAW_COUNT, ">", 0),
                AffineComparisonPredicate(
                    {
                        field.EQUILIBRIUM_CONSTRAINT_COUNT: 1.0,
                        field.STOICHIOMETRY_RANK: -1.0,
                    },
                    "==",
                    0.0,
                ),
            ),
        ),
    )


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
    ``IntegrationDriver`` because it modifies state between steps.

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
        max_rejections: int = 20,
    ) -> None:
        if eps_deactivate <= eps_activate:
            raise ValueError("eps_deactivate must exceed eps_activate for hysteresis.")
        self._rhs = rhs
        self._integrator = integrator
        self._inner = inner
        self._eps_activate = eps_activate
        self._eps_deactivate = eps_deactivate
        self._eps_grad = eps_grad
        self._max_rejections = max_rejections

        self._n_eligible = rhs.constraint_basis.shape[0]

        # Diagnostic event logs.
        self.activation_events: list[tuple[float, frozenset[int]]] = []
        self.deactivation_events: list[tuple[float, frozenset[int]]] = []
        self.nse_events: list[tuple[float, frozenset[int]]] = []
        self.rejection_reasons: list[str] = []
        self.domain_violations: list[DomainViolation] = []
        self.domain_rejection_step_sizes: list[float] = []
        self.domain_limited_step_sizes: list[float] = []
        self.rejected_steps = 0

    def advance(
        self,
        u0: Tensor,
        t0: float,
        t_end: float,
        *,
        initial_active: frozenset[int] | None = None,
        stop_at_nse: bool = False,
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
        stop_at_nse:
            Return as soon as all independent equilibrium constraints activate
            and the state has been projected onto the NSE manifold.
        """
        active: frozenset[int] = (
            initial_active if initial_active is not None else frozenset()
        )
        state = ODEState(t0, u0, active_constraints=active)
        dt = self._inner.suggest(state, accepted=True)
        rejections = 0

        while state.t < t_end:
            dt_try = min(dt, t_end - state.t)
            dt_try = self._limit_step_to_domain(state, dt_try)
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
            error_accepted = self._inner.accept(candidate)
            domain_check = check_state_domain(self._rhs, candidate.u)
            accepted = error_accepted and domain_check.accepted
            if not accepted:
                rejections += 1
                self.rejected_steps += 1
                if rejections > self._max_rejections:
                    raise RuntimeError(
                        "constraint-aware step exceeded rejection limit."
                    )
                if not error_accepted:
                    self.rejection_reasons.append("error")
                    dt = self._inner.suggest(candidate, accepted=False)
                    continue
                self.rejection_reasons.append("domain")
                assert domain_check.violation is not None
                self.domain_violations.append(domain_check.violation)
                self.domain_rejection_step_sizes.append(dt_try)
                dt = dt_try * self._inner.factor_min
                continue
            rejections = 0
            dt = self._inner.suggest(candidate, accepted=True)

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
            in_nse = self._n_eligible > 0 and len(new_active) == self._n_eligible
            if newly_on:
                self.activation_events.append((candidate.t, newly_on))
                if in_nse:
                    u_proj = solve_nse(
                        self._rhs, u_proj, candidate.t, eps=self._eps_grad
                    )
                    self.nse_events.append((candidate.t, new_active))
                else:
                    u_proj = _consistent_init(
                        self._rhs, u_proj, new_active, candidate.t, self._eps_grad
                    )
            if newly_off:
                self.deactivation_events.append((candidate.t, newly_off))

            state = candidate._replace(u=u_proj, active_constraints=new_active)
            if stop_at_nse and in_nse:
                return state
            if candidate.err == 0.0 and state.active_constraints:
                growth = getattr(self._inner, "factor_max", 1.0)
                dt = min(t_end - state.t, dt_try * growth)

        return state

    def _limit_step_to_domain(self, state: ODEState, dt: float) -> float:
        limit = predict_domain_step_limit(self._rhs, state.t, state.u)
        if limit is None or limit <= 0.0 or limit >= dt:
            return dt
        self.domain_limited_step_sizes.append(limit)
        return limit


__all__ = [
    "ConstraintAwareController",
    "NuclearStatisticalEquilibriumSolver",
    "build_constraint_gradients",
    "nse_root_relation",
    "reaction_network_coverage_regions",
    "solve_nse",
]
