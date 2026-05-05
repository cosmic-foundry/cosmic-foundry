"""Nordsieck-form linear multistep integrators: BDF and Adams-Moulton families.

The Nordsieck vector z_n stores the scaled Taylor expansion of y at t_n:

    z_n[j] = h^j / j! · y^(j)(t_n)   for j = 0, 1, …, q

where h is the step size baked into the state.  Prediction advances z_n to
t_{n+1} via the (q+1) × (q+1) Pascal matrix P[i,j] = C(j,i) for j ≥ i.
The corrector then applies the correction vector l component-wise to the
predicted Nordsieck vector after solving one implicit equation.

Both families share the same correction structure:

    z_new[i] = z_pred[i] + l[i] · δ,   δ = h·f(t_{n+1}, y_{n+1}) − z_pred[1]

The implicit equation for y_{n+1} is:

    y − β₀ h f(t_{n+1}, y) = z_pred[0] − β₀ z_pred[1]

and l[1] = 1 always (the derivative slot is updated to h·f exactly).

BDF family (stiff problems)
---------------------------
    β₀ = 1 / H_q,   H_q = Σ_{k=1}^{q} 1/k
    l[j] = β₀ · e_{q−j}(1, …, q) / q!   for j = 0, …, q

where e_k is the k-th elementary symmetric polynomial.  The implicit equation
is solved by Newton iteration using the Jacobian from WithJacobianRHSProtocol.

Adams-Moulton family (non-stiff problems)
-----------------------------------------
    β₀ = (1/(q−1)!) · ∫₀¹ ∏_{k=0}^{q−2} (s+k) ds
    l[j] = coefficient of ξ^j in ∫₀^ξ ∏_{k=1}^{q−1} (s+k) / (q−1)! ds

giving l = [β₀, 1, 1/2], [β₀, 1, 3/4, 1/6], [β₀, 1, 11/12, 1/3, 1/24], …
(Gear 1971 Table 11.2; l[q] = 1/q! always).  The implicit equation is
solved by fixed-point iteration, so no Jacobian is required; this family
satisfies plain RHSProtocol and is preferred for non-stiff problems.
"""

from __future__ import annotations

from math import comb, factorial  # comb used in _pascal_predict

import sympy

from cosmic_foundry.computation.solvers._root_execution import solve_root_relation
from cosmic_foundry.computation.solvers.newton_root_solver import RootRelation
from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators.implicit import WithJacobianRHSProtocol
from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    RHSProtocol,
)
from cosmic_foundry.computation.time_integrators.runge_kutta import (
    RungeKuttaIntegrator as _RungeKuttaIntegrator,
)
from cosmic_foundry.computation.time_integrators.stiffness import FamilyName

_bootstrap_rk = _RungeKuttaIntegrator(6)

_FP_MAX_ITER = 50
_FP_TOL = 1e-12


class NordsieckHistory:
    """Nordsieck vector and the step size for which it is scaled.

    z[j] = h^j / j! · y^(j)(t) for j = 0, …, q.  Storing h alongside z
    makes prediction a pure Pascal-matrix multiply independent of the next
    step size.  The derivative slot z[1] equals h · f(t, y) exactly after
    each accepted step.

    Each slot carries a scaled derivative of the solution: z[0] is y itself,
    z[1] is h times the velocity, z[2] is h²/2 times the acceleration, and
    so on.  Keeping h here rather than passing it separately lets the
    integrator rescale z cheaply when the step size changes.

    Parameters
    ----------
    h:
        Step size for which z is scaled.
    z:
        Nordsieck vector as a tuple of q+1 Tensors; z[j] ∈ ℝⁿ.
    """

    __slots__ = ("h", "z")

    def __init__(self, h: float, z: tuple[Tensor, ...]) -> None:
        self.h = h
        self.z = z

    @property
    def q(self) -> int:
        """Method order (len(z) − 1)."""
        return len(self.z) - 1

    @property
    def u(self) -> Tensor:
        """Current solution vector (z[0])."""
        return self.z[0]

    def rescale_step(self, h_new: float) -> NordsieckHistory:
        """Return this history rescaled to ``h_new``.

        Slot ``j`` is multiplied by ``(h_new / h)^j``; z[0] is unchanged.
        """
        if h_new <= 0.0:
            raise ValueError("Nordsieck step size must be positive.")
        if h_new == self.h:
            return self
        r = h_new / self.h
        return NordsieckHistory(
            h_new,
            tuple(self.z[j] * (r**j) for j in range(self.q + 1)),
        )

    def change_order(self, q_new: int) -> NordsieckHistory:
        """Return this history at order ``q_new``.

        Lowering order truncates the highest derivative slots.  Raising order
        pads new slots with zero.
        """
        if q_new < 0:
            raise ValueError("Nordsieck order must be non-negative.")
        q_old = self.q
        if q_new == q_old:
            return self
        if q_new < q_old:
            return NordsieckHistory(self.h, self.z[: q_new + 1])

        zero = Tensor.zeros(*self.u.shape, backend=self.u.backend)
        padding = tuple(zero for _ in range(q_new - q_old))
        return NordsieckHistory(self.h, self.z + padding)


def _pascal_predict(z: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Advance Nordsieck vector by one step via Pascal-triangle multiply.

    z_pred[i] = Σ_{j=i}^{q} C(j, i) · z[j], which is the i-th component
    of P · z where P is the (q+1) × (q+1) upper-triangular Pascal matrix.
    For a polynomial y of degree ≤ q this is exact; for general y it is
    the Adams-type explicit predictor.
    """
    q = len(z) - 1
    z_pred: list[Tensor] = []
    for i in range(q + 1):
        acc = z[i]
        for j in range(i + 1, q + 1):
            acc = acc + comb(j, i) * z[j]
        z_pred.append(acc)
    return tuple(z_pred)


def bdf_corrector_root_relation(
    rhs: WithJacobianRHSProtocol,
    z_pred: tuple[Tensor, ...],
    t_new: float,
    h: float,
    beta0: float,
) -> RootRelation:
    """Return the BDF corrector root relation for one Nordsieck step."""
    y_initial = z_pred[0]
    n = y_initial.shape[0]
    backend = y_initial.backend
    rhs_bdf = z_pred[0] - beta0 * z_pred[1]

    def residual(y: Tensor) -> Tensor:
        return y - (beta0 * h) * rhs(t_new, y) - rhs_bdf

    def jacobian(y: Tensor) -> Tensor:
        return Tensor.eye(n, backend=backend) - (beta0 * h) * rhs.jacobian(t_new, y)

    return RootRelation(residual, jacobian, y_initial)


class _BDFFamily:
    """Parametric BDF coefficient provider for orders 1 through q_max.

    BDF-q advances the ODE using the implicit linear multistep formula
    y_{n+1} − β₀ h f(t_{n+1}, y_{n+1}) = Σ α_j y_{n−j+1} (j ≥ 1), where
    β₀ = 1/H_q is determined by the harmonic number H_q = Σ 1/k.

    In Nordsieck form the only coefficients needed per step are β₀ and the
    correction vector l; both are precomputed as exact rationals at
    construction time.  The Newton corrector for each step then requires only
    β₀; l is applied after convergence to update the full history vector.

    Use this family whenever the problem is stiff and a Jacobian is available.
    Adams-type methods (Phase 7b) are preferred for non-stiff problems because
    they achieve the same order at lower cost (no Jacobian required).

    Parameters
    ----------
    q_max:
        Maximum supported order.  Default 6 (all A-stable BDF orders).
    """

    def __init__(self, q_max: int = 6) -> None:
        self._q_max = q_max
        self._l_sym: dict[int, list[sympy.Rational]] = {}
        self._l_f: dict[int, list[float]] = {}
        self._beta0_f: dict[int, float] = {}
        for q in range(1, q_max + 1):
            l_sym, beta0_sym = self._compute_l(q)
            self._l_sym[q] = l_sym
            self._l_f[q] = [float(v) for v in l_sym]
            self._beta0_f[q] = float(beta0_sym)

    @staticmethod
    def _compute_l(q: int) -> tuple[list[sympy.Rational], sympy.Rational]:
        """Return the BDF-q l-vector and β₀ as exact sympy rationals.

        Uses l[j] = β₀ · e_{q−j}(1, …, q) / q! for all j, which gives
        l[0] = β₀, l[1] = 1 (verified by e_{q−1}/q! = H_q, so β₀·H_q = 1).
        """
        H_q = sum(sympy.Rational(1, k) for k in range(1, q + 1))
        beta0: sympy.Rational = sympy.Rational(1) / H_q

        # e[k] = e_k(1, 2, …, q): elementary symmetric polynomial
        e: list[sympy.Integer] = [sympy.Integer(0)] * (q + 1)
        e[0] = sympy.Integer(1)
        for x in range(1, q + 1):
            for k in range(min(x, q), 0, -1):
                e[k] = e[k] + x * e[k - 1]

        q_fact = sympy.Integer(factorial(q))
        lvec = [beta0 * e[q - j] / q_fact for j in range(q + 1)]
        return lvec, beta0

    def l_vector(self, q: int) -> list[float]:
        """BDF-q correction vector as Python floats."""
        return self._l_f[q]

    def l_vector_sym(self, q: int) -> list[sympy.Rational]:
        """BDF-q correction vector as exact sympy rationals."""
        return self._l_sym[q]

    def beta0(self, q: int) -> float:
        """β₀ for BDF-q as a Python float."""
        return self._beta0_f[q]

    @property
    def q_max(self) -> int:
        """Maximum supported order."""
        return self._q_max


class _AdamsFamily:
    """Parametric Adams-Moulton coefficient provider for orders 1 through q_max.

    Adams-Moulton of order q is the implicit corrector:

        y_{n+1} = y_n + h · (β₀ f_{n+1} + β₁ f_n + … + β_{q-1} f_{n-q+2})

    In Nordsieck form the only per-step quantities needed are β₀ and the
    correction vector l, both precomputed as exact rationals.

    β₀ is the leading Adams-Moulton coefficient:

        β₀ = (1/(q−1)!) · ∫₀¹ ∏_{k=0}^{q−2} (s+k) ds

    which evaluates to 1, 1/2, 5/12, 3/8, 251/720, … for q = 1, 2, 3, 4, 5, …

    The l-vector satisfies l[0] = β₀, l[1] = 1 (always), and:

        l[j] = coefficient of ξ^j in ∫₀^ξ ∏_{k=1}^{q−1} (s+k) / (q−1)! ds

    for j = 1, …, q.  This integral polynomial generates l[j] = 1, 1/2, 3/4,
    1/6, 11/12, 1/3, 1/24, … for increasing q and j, matching Gear (1971)
    Table 11.2.  Explicitly: l[q] = 1/q! always, and l[2] = H_{q−1}/2 where
    H_k is the k-th harmonic number.

    In plain terms: the l-vector encodes both the one-step prediction-error
    correction (l[1]=1 always) and the inter-step consistency requirement —
    after the correction, the Nordsieck history must correctly seed the
    Pascal prediction for the next step.  The integral formula accounts for
    both constraints; using only the prediction error gives wrong values for
    j ≥ 2 that make the higher Nordsieck slots unstable.

    Use this family for non-stiff problems.  The implicit equation is solved
    by fixed-point iteration — no Jacobian is required (plain RHSProtocol).
    For stiff problems use BDF, which solves via Newton iteration.

    Parameters
    ----------
    q_max:
        Maximum supported order.  Default 12 (AM1–AM12).
    """

    def __init__(self, q_max: int = 12) -> None:
        self._q_max = q_max
        self._l_sym: dict[int, list[sympy.Rational]] = {}
        self._l_f: dict[int, list[float]] = {}
        self._beta0_f: dict[int, float] = {}
        for q in range(1, q_max + 1):
            l_sym, beta0_sym = self._compute_l(q)
            self._l_sym[q] = l_sym
            self._l_f[q] = [float(v) for v in l_sym]
            self._beta0_f[q] = float(beta0_sym)

    @staticmethod
    def _compute_l(q: int) -> tuple[list[sympy.Rational], sympy.Rational]:
        """Return the Adams-Moulton-q l-vector and β₀ as exact sympy rationals.

        β₀ = (1/(q−1)!) · ∫₀¹ ∏_{k=0}^{q−2} (s+k) ds

        l[0] = β₀
        l[j] = coefficient of ξ^j in ∫₀^ξ ∏_{k=1}^{q−1} (s+k) / (q−1)! ds,
               for j = 1, …, q

        The generating integral uses roots at s = −1, …, −(q−1), one shifted
        relative to the β₀ integrand (roots at 0, …, −(q−2)).  The shift
        ensures l[q] = 1/q! and satisfies the inter-step consistency
        condition that the l-vector corrector in Nordsieck form produces.
        """
        s = sympy.Symbol("s")
        xi = sympy.Symbol("xi")

        # beta0: ∫₀¹ s(s+1)⋯(s+q−2) / (q−1)! ds
        integrand_beta = sympy.Integer(1)
        for k in range(q - 1):
            integrand_beta *= s + k
        beta0: sympy.Rational = sympy.integrate(
            integrand_beta, (s, 0, 1)
        ) / sympy.factorial(q - 1)

        # l[1..q]: from ∫₀^ξ (s+1)(s+2)⋯(s+q−1) / (q−1)! ds
        integrand_l = sympy.Integer(1)
        for k in range(1, q):
            integrand_l *= s + k
        integrand_l = sympy.expand(integrand_l) / sympy.factorial(q - 1)
        L_poly = sympy.expand(sympy.integrate(integrand_l, (s, 0, xi)))

        lvec: list[sympy.Rational] = [beta0]
        for j in range(1, q + 1):
            lvec.append(L_poly.coeff(xi, j))
        return lvec, beta0

    def l_vector(self, q: int) -> list[float]:
        """Adams-Moulton-q correction vector as Python floats."""
        return self._l_f[q]

    def l_vector_sym(self, q: int) -> list[sympy.Rational]:
        """Adams-Moulton-q correction vector as exact sympy rationals."""
        return self._l_sym[q]

    def beta0(self, q: int) -> float:
        """β₀ for Adams-Moulton-q as a Python float."""
        return self._beta0_f[q]

    @property
    def q_max(self) -> int:
        """Maximum supported order."""
        return self._q_max


class _MultistepCoefficients:
    def __init__(self) -> None:
        self.bdf = _BDFFamily(q_max=6)
        self.adams = _AdamsFamily(q_max=6)


_COEFFS = _MultistepCoefficients()


class MultistepIntegrator:
    """Fixed-order linear multistep integrator in Nordsieck form.

    Supports two corrector families selected by the family argument:

    BDF (stiff problems)
        Each step solves y − β₀ h f(t_{n+1}, y) = z_pred[0] − β₀ z_pred[1]
        via Newton iteration, requiring WithJacobianRHSProtocol.

    Adams (non-stiff problems)
        Each step solves the same equation via fixed-point iteration,
        requiring only plain RHSProtocol (no Jacobian).

    Both paths share the same prediction (Pascal matrix) and history update
    (l-vector correction).  The corrector family alone determines whether
    Newton or fixed-point iteration is used; no other interface changes.

    In plain terms: the prediction step extrapolates the polynomial history
    to t_{n+1}; the correction step adjusts y_{n+1} until the implicit
    equation is satisfied; the l-vector distributes that adjustment across
    all Nordsieck slots so the history stays self-consistent.

    Use init_state() to create the initial multistep state by bootstrapping
    q RK4 steps.  BDF init uses one Jacobian evaluation to fill higher
    Nordsieck slots; Adams init uses backward differences of the bootstrap
    function values (no Jacobian required).

    Parameters
    ----------
    family:
        ``"bdf"`` or ``"adams"``.
    q:
        Method order (1 ≤ q ≤ 6).
    """

    def __init__(self, family: FamilyName | str, q: int) -> None:
        family = FamilyName(family)
        fam = _COEFFS.bdf if family == "bdf" else _COEFFS.adams
        if not 1 <= q <= fam.q_max:
            raise ValueError(f"q={q} outside [1, {fam.q_max}]")
        self._family_name = family
        self._q = q
        self._l = fam.l_vector(q)
        self._beta0 = fam.beta0(q)

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._q

    @property
    def family(self) -> FamilyName:
        """Corrector family induced by the selected step descriptor."""
        return self._family_name

    def init_state(
        self,
        rhs: RHSProtocol,
        t0: float,
        u0: Tensor,
        dt: float,
    ) -> ODEState:
        """Bootstrap q RK4 steps and initialize the Nordsieck vector.

        For ``"bdf"``: initializes z[j] via the Jacobian-based Taylor recursion
        z[j] = dt^j/j! · J^{j-1} · f evaluated at (t_q, y_q).  Requires
        WithJacobianRHSProtocol; raises AttributeError at runtime otherwise.

        For ``"adams"``: initializes z[j] via a corrected backward-difference
        formula using the bootstrap function values:

            z[j] = (dt / j!) · (∇^{j-1} f_q + (j−1)/2 · ∇^j f_q)
            ∇^k f_q = Σ_{i=0}^{k} (−1)^i C(k,i) f_{q−i}

        The plain ∇^{j-1} f_q approximates h^{j-1} y^{(j)} with error
        O(h^j); adding (j−1)/2 · ∇^j f_q cancels the leading error term,
        reducing the initialization error from O(h^{j+1}) to O(h^{j+2}).
        This keeps the bootstrap error one order below the method's global
        error for all q, enabling AM1–AM6 to achieve their declared orders.

        Both paths take q RK4 steps from (t0, u0) and arrive at (t_q, y_q)
        with t_q = t0 + q·dt.

        Parameters
        ----------
        rhs:
            ODE right-hand side.  Must satisfy WithJacobianRHSProtocol for
            ``"bdf"``; plain RHSProtocol suffices for ``"adams"``.
        t0:
            Initial time.
        u0:
            Initial state vector.
        dt:
            Step size for the bootstrap and all subsequent steps.
        """
        if self._family_name == "bdf":
            return self._init_state_bdf(rhs, t0, u0, dt)  # type: ignore[arg-type]
        return self._init_state_adams(rhs, t0, u0, dt)

    def _init_state_bdf(
        self,
        rhs: WithJacobianRHSProtocol,
        t0: float,
        u0: Tensor,
        dt: float,
    ) -> ODEState:
        q = self._q
        rk_state = ODEState(t0, u0)
        for _ in range(q):
            rk_state = _bootstrap_rk.step(rhs, rk_state, dt)
        t_q = rk_state.t
        y_q = rk_state.u

        J = rhs.jacobian(t_q, y_q)
        f_vec = rhs(t_q, y_q)
        z: list[Tensor] = [y_q, dt * f_vec]
        f_deriv = f_vec
        for j in range(2, q + 1):
            f_deriv = J @ f_deriv
            z.append(f_deriv * (dt**j / factorial(j)))

        nh = NordsieckHistory(dt, tuple(z))
        return ODEState(t_q, y_q, dt, 0.0, nh)

    def _init_state_adams(
        self,
        rhs: RHSProtocol,
        t0: float,
        u0: Tensor,
        dt: float,
    ) -> ODEState:
        q = self._q
        rk_states = [ODEState(t0, u0)]
        for _ in range(q):
            rk_states.append(_bootstrap_rk.step(rhs, rk_states[-1], dt))
        t_q = rk_states[-1].t
        y_q = rk_states[-1].u

        # Function values at t0, t1, ..., tq for backward-difference init.
        f_vals = [rhs(s.t, s.u) for s in rk_states]
        f_q = f_vals[q]
        z: list[Tensor] = [y_q, dt * f_q]
        for j in range(2, q + 1):
            # nabla^{j-1} f_q = sum_{i=0}^{j-1} (-1)^i C(j-1, i) f_{q-i}
            nabla: Tensor = f_vals[q]
            for i in range(1, j):
                nabla = nabla + ((-1) ** i * comb(j - 1, i)) * f_vals[q - i]
            # One-order accuracy improvement: nabla^{j-1} approximates h^{j-1}y^{(j)}
            # with error -(j-1)/2 * h^j y^{(j+1)} + O(h^{j+1}).  Adding (j-1)/2 times
            # the next backward difference nabla^j (which approximates h^j y^{(j+1)})
            # cancels the leading error term, reducing init error from O(h^{j+1}) to
            # O(h^{j+2}).  nabla^j f_q requires f_{q-j} = f_vals[q-j], which is always
            # available since j <= q and the bootstrap provides f_0,...,f_q.
            nabla_j: Tensor = f_vals[q]
            for i in range(1, j + 1):
                nabla_j = nabla_j + ((-1) ** i * comb(j, i)) * f_vals[q - i]
            nabla = nabla + ((j - 1) / 2) * nabla_j
            z.append(nabla * (dt / factorial(j)))

        nh = NordsieckHistory(dt, tuple(z))
        return ODEState(t_q, y_q, dt, 0.0, nh)

    def step(
        self,
        rhs: RHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        """Advance state by one step of size dt.

        Rescales z if dt ≠ state.history.h, predicts via Pascal, corrects via
        Newton (BDF) or fixed-point iteration (Adams), then updates
        the full Nordsieck vector with l · f_delta where
        f_delta = h · f(t_{n+1}, y_{n+1}) − z_pred[1].

        Returns a new ODEState at t + dt.
        """
        if self._family_name == "bdf":
            return self._step_bdf(rhs, state, dt)  # type: ignore[arg-type]
        return self._step_adams(rhs, state, dt)

    def _step_bdf(
        self,
        rhs: WithJacobianRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        nh: NordsieckHistory = state.history
        if nh.q != self._q:
            raise ValueError(
                f"Nordsieck history order {nh.q} does not match integrator "
                f"order {self._q}."
            )
        nh = nh.rescale_step(dt)
        t, h, z = state.t, nh.h, nh.z
        q = nh.q

        z_pred = _pascal_predict(z)
        t_new = t + h
        beta0 = self._beta0
        y = solve_root_relation(
            bdf_corrector_root_relation(rhs, z_pred, t_new, h, beta0)
        )

        fy = rhs(t_new, y)
        f_delta = h * fy - z_pred[1]
        lvec = self._l
        z_new = tuple(z_pred[i] + lvec[i] * f_delta for i in range(q + 1))
        return ODEState(t_new, y, h, 0.0, NordsieckHistory(h, z_new))

    def _step_adams(
        self,
        rhs: RHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        nh: NordsieckHistory = state.history
        if nh.q != self._q:
            raise ValueError(
                f"Nordsieck history order {nh.q} does not match integrator "
                f"order {self._q}."
            )
        nh = nh.rescale_step(dt)
        t, h, z = state.t, nh.h, nh.z
        q = nh.q

        z_pred = _pascal_predict(z)
        t_new = t + h
        beta0 = self._beta0

        y = z_pred[0]
        for _ in range(_FP_MAX_ITER):
            fy = rhs(t_new, y)
            y_new = z_pred[0] + beta0 * (h * fy - z_pred[1])
            delta_norm = float(norm(y_new - y))
            y = y_new
            if delta_norm < _FP_TOL * (1.0 + float(norm(y))):
                break

        fy = rhs(t_new, y)
        f_delta = h * fy - z_pred[1]
        lvec = self._l
        z_new = tuple(z_pred[i] + lvec[i] * f_delta for i in range(q + 1))
        return ODEState(t_new, y, h, 0.0, NordsieckHistory(h, z_new))


__all__ = [
    "MultistepIntegrator",
    "NordsieckHistory",
    "bdf_corrector_root_relation",
]
