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
    l[j] = C(q, j) / (q−j+1)   for j = 1, …, q

The l-vector derivation: δ = h^{q+1}/q! · y^{(q+1)} + O(h^{q+2}), and
requiring z_new[j] = h^j/j! · y^{(j)}(t_{n+1}) + O(h^{q+2}) for j ≥ 1
gives l[j] = q! / (j! · (q−j+1)!) = C(q,j)/(q−j+1).  The implicit equation
is solved by fixed-point iteration, so no Jacobian is required; this family
satisfies plain RHSProtocol and is preferred for non-stiff problems.

Named instances
---------------
bdf1–bdf4          —  BDF orders 1–4 (L/A/A(α)-stable)
adams_moulton1–am4 —  Adams-Moulton orders 1–4 (non-stiff, no Jacobian)
"""

from __future__ import annotations

from math import comb, factorial  # comb used in _pascal_predict

import sympy

from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.tensor import Tensor, norm
from cosmic_foundry.computation.time_integrators.implicit import WithJacobianRHSProtocol
from cosmic_foundry.computation.time_integrators.integrator import RKState
from cosmic_foundry.computation.time_integrators.runge_kutta import rk4 as _rk4

_LU = LUFactorization()
_NEWTON_MAX_ITER = 50
_NEWTON_TOL = 1e-12


class NordsieckState:
    """State for Nordsieck-form linear multistep integrators.

    z[j] = h^j / j! · y^(j)(t) for j = 0, …, q.  Storing h in the state
    makes prediction a pure Pascal-matrix multiply independent of the next
    step size.  The derivative slot z[1] equals h · f(t, y) exactly after
    each accepted step.

    In plain terms: each slot of z carries a scaled derivative of the
    solution — the zeroth slot is y itself, the first is h times the
    velocity, the second is h²/2 times the acceleration, and so on.
    Keeping h in the state rather than as an argument to step() lets the
    integrator rescale z cheaply when the step size changes.

    Parameters
    ----------
    t:
        Current time.
    h:
        Step size for which z is scaled.
    z:
        Nordsieck vector as a tuple of q+1 Tensors; z[j] ∈ ℝⁿ.
    """

    __slots__ = ("t", "h", "z")

    def __init__(self, t: float, h: float, z: tuple[Tensor, ...]) -> None:
        self.t = t
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


class BDFFamily:
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


class AdamsFamily:
    """Parametric Adams-Moulton coefficient provider for orders 1 through q_max.

    Adams-Moulton of order q is the implicit corrector:

        y_{n+1} = y_n + h · (β₀ f_{n+1} + β₁ f_n + … + β_{q-1} f_{n-q+2})

    In Nordsieck form the only per-step quantities needed are β₀ and the
    correction vector l, both precomputed as exact rationals.

    β₀ is the leading Adams-Moulton coefficient:

        β₀ = (1/(q−1)!) · ∫₀¹ ∏_{k=0}^{q−2} (s+k) ds

    which evaluates to 1, 1/2, 5/12, 3/8, 251/720, … for q = 1, 2, 3, 4, 5, …

    The l-vector satisfies l[1] = 1 (always) and:

        l[j] = C(q, j) / (q−j+1)   for j = 1, …, q
        l[0] = β₀

    Derivation: δ = h·f_{n+1} − z_pred[1] = h^{q+1}/q! · y^{(q+1)} + O(h^{q+2}).
    Requiring z_new[j] = h^j/j! · y^{(j)}(t_{n+1}) + O(h^{q+2}) for j = 1, …, q
    gives l[j] · (h^{q+1}/q!) = h^{q+1}/(j!(q−j+1)!), hence l[j] = C(q,j)/(q−j+1).

    In plain terms: the prediction step (Pascal multiply) is exact for
    polynomials of degree ≤ q; the correction l·δ patches the O(h^{q+1})
    error introduced by the leading-order Taylor term that the predictor misses.

    Use this family for non-stiff problems.  The implicit equation is solved
    by fixed-point iteration — no Jacobian is required (plain RHSProtocol).
    For stiff problems use BDFFamily, which solves via Newton iteration.

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
        l[j] = C(q, j) / (q−j+1)   for j = 1, …, q
        """
        s = sympy.Symbol("s")
        integrand = sympy.Integer(1)
        for k in range(q - 1):
            integrand *= s + k
        beta0: sympy.Rational = sympy.integrate(integrand, (s, 0, 1)) / sympy.factorial(
            q - 1
        )

        lvec: list[sympy.Rational] = [beta0]
        for j in range(1, q + 1):
            lvec.append(sympy.binomial(q, j) / sympy.Integer(q - j + 1))
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


class NordsieckIntegrator:
    """Fixed-order BDF integrator in Nordsieck form.

    Each step predicts via the Pascal matrix then corrects by Newton
    iteration on the BDF implicit equation.  The correction vector l
    updates the full Nordsieck history after convergence so that z[1]
    equals h · f(t_{n+1}, y_{n+1}) exactly.

    A Jacobian is required (WithJacobianRHSProtocol) because BDF is
    unconditionally implicit; every stage inverts (I − β₀ h J).

    In plain terms: each BDF step first extrapolates the current
    polynomial history to t_{n+1} (the prediction), then corrects the
    predicted y by iterating until the BDF implicit equation is satisfied,
    and finally updates all higher derivative slots so the history stays
    consistent.

    Use init_state() to create the initial NordsieckState from (t₀, y₀, h)
    by bootstrapping q RK4 steps.

    Parameters
    ----------
    family:
        BDFFamily instance supplying l and β₀ coefficients.
    q:
        Method order (1 ≤ q ≤ family.q_max).
    """

    def __init__(self, family: BDFFamily, q: int) -> None:
        if not 1 <= q <= family.q_max:
            raise ValueError(f"q={q} outside [1, {family.q_max}]")
        self._family = family
        self._q = q
        self._l = family.l_vector(q)
        self._beta0 = family.beta0(q)

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._q

    def init_state(
        self,
        rhs: WithJacobianRHSProtocol,
        t0: float,
        u0: Tensor,
        dt: float,
    ) -> NordsieckState:
        """Bootstrap q RK4 steps and initialize the Nordsieck vector.

        Takes q RK4 steps from (t0, u0) with step size dt, arriving at
        (t_q, y_q) with t_q = t0 + q*dt.  Then initializes z[j] using
        the J-based Taylor recursion:

            z[0] = y_q
            z[1] = dt * f(t_q, y_q)
            z[j] = dt^j / j! * J^{j-1} * f   for j = 2, ..., q

        where J = df/dy(t_q, y_q).  For autonomous linear ODEs this is
        exact; for nonlinear problems the error in z[j] is O(dt^{j+1})
        from missing Hessian terms, which is O(dt^{q+1}) overall and does
        not degrade the BDF-q convergence rate.

        Parameters
        ----------
        rhs:
            ODE right-hand side; Jacobian evaluated once at (t_q, y_q).
        t0:
            Initial time.
        u0:
            Initial state vector.
        dt:
            Step size for the bootstrap and all subsequent steps.
        """
        q = self._q
        rk_state = RKState(t0, u0)
        for _ in range(q):
            rk_state = _rk4.step(rhs, rk_state, dt)
        t_q = rk_state.t
        y_q = rk_state.u

        J = rhs.jacobian(t_q, y_q)
        f_vec = rhs(t_q, y_q)
        z: list[Tensor] = [y_q, dt * f_vec]
        f_deriv = f_vec
        for j in range(2, q + 1):
            f_deriv = J @ f_deriv
            z.append(f_deriv * (dt**j / factorial(j)))

        return NordsieckState(t_q, dt, tuple(z))

    def step(
        self,
        rhs: WithJacobianRHSProtocol,
        state: NordsieckState,
        dt: float,
    ) -> NordsieckState:
        """Advance state by one step of size dt.

        Rescales z if dt ≠ state.h, predicts via Pascal, then corrects via
        Newton iteration on y − β₀ h f(t_{n+1}, y) = z^(0)[0] − β₀ z^(0)[1].
        After convergence updates the full Nordsieck vector with l · f_delta,
        where f_delta = h · f(t_{n+1}, y_{n+1}) − z^(0)[1].

        Returns a new NordsieckState at t + dt.
        """
        t, h, z = state.t, state.h, state.z
        q = state.q
        n = z[0].shape[0]
        backend = z[0].backend

        if dt != h:
            r = dt / h
            z = tuple(z[j] * (r**j) for j in range(q + 1))
            h = dt

        z_pred = _pascal_predict(z)
        t_new = t + h
        beta0 = self._beta0
        rhs_bdf = z_pred[0] - beta0 * z_pred[1]

        y = z_pred[0]
        for _ in range(_NEWTON_MAX_ITER):
            fy = rhs(t_new, y)
            F = y - (beta0 * h) * fy - rhs_bdf
            if float(norm(F)) < _NEWTON_TOL * (1.0 + float(norm(y))):
                break
            J = rhs.jacobian(t_new, y)
            M = Tensor.eye(n, backend=backend) - (beta0 * h) * J
            delta = _LU.factorize(M).solve(Tensor.zeros(n, backend=backend) - F)
            y = y + delta
            if float(norm(delta)) < _NEWTON_TOL * (1.0 + float(norm(y))):
                break

        fy = rhs(t_new, y)
        f_delta = h * fy - z_pred[1]
        lvec = self._l
        z_new = tuple(z_pred[i] + lvec[i] * f_delta for i in range(q + 1))
        return NordsieckState(t_new, h, z_new)


# ---------------------------------------------------------------------------
# Named instances
# ---------------------------------------------------------------------------

bdf_family = BDFFamily(q_max=6)

bdf1 = NordsieckIntegrator(bdf_family, q=1)
bdf2 = NordsieckIntegrator(bdf_family, q=2)
bdf3 = NordsieckIntegrator(bdf_family, q=3)
bdf4 = NordsieckIntegrator(bdf_family, q=4)


__all__ = [
    "AdamsFamily",
    "BDFFamily",
    "NordsieckIntegrator",
    "NordsieckState",
    "bdf1",
    "bdf2",
    "bdf3",
    "bdf4",
    "bdf_family",
]
