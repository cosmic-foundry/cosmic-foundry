"""Reaction network RHS with explicit stoichiometric structure."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cosmic_foundry.computation.decompositions.lu_factorization import LUFactorization
from cosmic_foundry.computation.tensor import Tensor, einsum
from cosmic_foundry.computation.time_integrators.domains import NonnegativeStateDomain

_LU = LUFactorization()


def _left_null_space(s_tensor: Tensor) -> list[list[float]]:
    """Return an orthonormal basis for null(Sᵀ) as row vectors.

    The left null space of S consists of vectors w satisfying w @ S = 0,
    equivalently the null space of Sᵀ.  Uses full SVD of Sᵀ via the
    tensor's backend; the threshold for a zero singular value is
    n·ε_machine·σ₀, where σ₀ is the largest singular value.
    """
    n_species, n_reactions = s_tensor.shape
    if n_reactions == 0 or n_species == 0:
        return []
    backend = s_tensor.backend
    s_t = einsum("ij->ji", s_tensor)  # (n_reactions, n_species)
    _, sigma_raw, vt_raw = backend.svd(s_t._value, s_t.shape, full_matrices=True)
    sigma = [float(x) for x in sigma_raw]
    k = len(sigma)
    tol = max(n_reactions, n_species) * np.finfo(float).eps * (sigma[0] if k else 1.0)
    rank = sum(1 for s in sigma if s > tol)
    result: list[list[float]] = [[float(x) for x in row] for row in vt_raw[rank:]]
    return result


def _independent_reaction_pairs(
    S_list: list[list[float]],
    n_conserved: int,
) -> list[int]:
    """Return indices of a maximal independent subset of reaction-pair constraints.

    Each pair j contributes a potential equilibrium constraint {r⁺ⱼ = r⁻ⱼ} that
    acts on species in the direction of the j-th column of S.  Independence is
    determined by Gram-Schmidt on those columns: add pair j if its column is not
    in the span of already-selected columns.  The maximum rank is
    n_species − n_conserved.
    """
    arr = np.array(S_list, dtype=float)
    n_species, n_reactions = arr.shape
    if n_reactions == 0:
        return []
    max_rank = n_species - n_conserved
    selected: list[int] = []
    basis: list[np.ndarray] = []
    for j in range(n_reactions):
        col = arr[:, j]
        if np.linalg.norm(col) < 1e-12:
            continue
        residual = col - sum((v @ col) * v for v in basis) if basis else col
        if np.linalg.norm(residual) > 1e-8:
            selected.append(j)
            basis.append(residual / np.linalg.norm(residual))
        if len(selected) >= max_rank:
            break
    return selected


def project_conserved(u: Tensor, basis: Tensor, targets: Tensor) -> Tensor:
    """Project u onto the conservation hyperplane {u : basis @ u = targets}.

    Returns the nearest point u′ satisfying basis @ u′ = targets by
    orthogonal projection:

        residual = basis @ u − targets          (shape: n_conserved)
        u′ = u − basisᵀ (basis basisᵀ)⁻¹ residual

    When u already satisfies the constraint the residual is zero and u is
    returned unchanged to floating-point precision.  The projection is
    idempotent and minimum-norm: u′ is the unique point on the hyperplane
    closest to u in the Euclidean norm.

    Parameters
    ----------
    u:
        State vector to project, shape (n_species,).
    basis:
        Conservation-law matrix, shape (n_conserved, n_species).  Rows are
        the left null-space vectors of the stoichiometry matrix S.
    targets:
        Target values basis @ u₀, shape (n_conserved,).
    """
    residual = basis @ u - targets
    gram = einsum("ij,kj->ik", basis, basis)
    delta = _LU.factorize(gram).solve(residual)
    correction = einsum("ij,i->j", basis, delta)
    return u - correction


class ReactionNetworkRHS:
    """RHS for a system of paired forward/reverse reactions with explicit stoichiometry.

    Let n = n_species and m = n_reactions (forward/reverse pairs).  The dynamics
    are du/dt = f(t, u) = S · (r⁺(t, u) − r⁻(t, u)), where S ∈ ℤⁿˣᵐ encodes the
    net species change per reaction (column j = net change when pair j fires once
    in the forward direction), r⁺ⱼ ≥ 0 is the forward rate, and r⁻ⱼ ≥ 0 is the
    reverse rate.  ``reverse_rate`` must be derived from ``forward_rate`` and
    thermodynamic data via the detailed balance relation — it is not a free
    parameter.  This guarantees the fully-equilibrated network recovers the correct
    thermodynamic fixed point: when all r⁺ⱼ = r⁻ⱼ, f(t, u) = 0 at the true
    thermodynamic equilibrium.

    The left null space of S spans the linear conservation laws: vectors w with
    w · S = 0, so that w · (du/dt) = 0 for all t, u.  These are computed at
    construction from S and stored as ``conservation_basis`` (shape n_conserved ×
    n_species); ``conservation_targets`` holds w · u₀ for each row w.

    ``constraint_basis`` records the indices of a maximal linearly independent
    subset of the m pairwise equilibrium conditions {r⁺ⱼ = r⁻ⱼ}.  The rank is
    at most n − n_conserved.  The ``ConstraintAwareController`` (introduced in
    phase F4) uses this to avoid activating redundant constraints that would make
    the constrained Newton system rank-deficient.

    In plain terms: define your reactions as forward/reverse rate pairs derived
    from thermodynamics, provide the stoichiometry matrix, and this class
    pre-computes what is conserved and which equilibrium conditions are
    independent.  The integrator can then use ``conservation_basis`` and
    ``constraint_basis`` at runtime without repeating that analysis.

    Satisfies ``WithJacobianRHSProtocol`` (``__call__`` + ``jacobian``).  A
    finite-difference Jacobian is used unless an analytical callable is supplied.

    Parameters
    ----------
    stoichiometry_matrix:
        S, shape (n_species, n_reactions).  Column j encodes the net species
        change when pair j fires once in the forward direction.
    forward_rate:
        r⁺(t, u) → Tensor of shape (n_reactions,), non-negative for u ≥ 0.
    reverse_rate:
        r⁻(t, u) → Tensor of shape (n_reactions,).  Must be derived from
        ``forward_rate`` via the detailed balance relation; not independently
        fitted.  The caller is responsible for this invariant.
    initial_state:
        u₀ used to set ``conservation_targets = conservation_basis @ u₀``.
    jac:
        Optional analytical Jacobian callable (t, u) → n × n Tensor.  When
        ``None``, forward finite differences are used.
    eps:
        Finite-difference step size when ``jac`` is ``None``.  Default 1e-7.
    """

    def __init__(
        self,
        stoichiometry_matrix: Tensor,
        forward_rate: Callable[[float, Tensor], Tensor],
        reverse_rate: Callable[[float, Tensor], Tensor],
        initial_state: Tensor,
        *,
        jac: Callable[[float, Tensor], Tensor] | None = None,
        eps: float = 1e-7,
    ) -> None:
        self._S = stoichiometry_matrix
        self._r_plus = forward_rate
        self._r_minus = reverse_rate
        self._jac_fn = jac
        self._eps = eps

        n_species, n_reactions = stoichiometry_matrix.shape
        self._state_domain = NonnegativeStateDomain(n_species)
        backend = initial_state.backend

        null_rows = _left_null_space(stoichiometry_matrix)
        n_conserved = len(null_rows)

        S_list = [
            [float(stoichiometry_matrix[i, j]) for j in range(n_reactions)]
            for i in range(n_species)
        ]

        if n_conserved > 0:
            self.conservation_basis: Tensor = Tensor(null_rows, backend=backend)
            targets = [
                sum(null_rows[i][k] * float(initial_state[k]) for k in range(n_species))
                for i in range(n_conserved)
            ]
            self.conservation_targets: Tensor = Tensor(targets, backend=backend)
        else:
            self.conservation_basis = Tensor([[0.0] * n_species], backend=backend)
            self.conservation_targets = Tensor([0.0], backend=backend)

        self.n_conserved: int = n_conserved

        pair_indices = _independent_reaction_pairs(S_list, n_conserved)
        self.constraint_basis = (
            Tensor([float(j) for j in pair_indices], backend=backend)
            if pair_indices
            else Tensor([0.0], backend=backend)[:0]
        )

    @property
    def stoichiometry_matrix(self) -> Tensor:
        """S, shape (n_species, n_reactions)."""
        return self._S

    @property
    def state_domain(self) -> NonnegativeStateDomain:
        """Valid state domain for reaction-network species abundances."""
        return self._state_domain

    def forward_rate(self, t: float, u: Tensor) -> Tensor:
        """Evaluate r⁺(t, u), shape (n_reactions,)."""
        return self._r_plus(t, u)

    def reverse_rate(self, t: float, u: Tensor) -> Tensor:
        """Evaluate r⁻(t, u), shape (n_reactions,)."""
        return self._r_minus(t, u)

    def __call__(self, t: float, u: Tensor) -> Tensor:
        """Evaluate du/dt = S · (r⁺(t, u) − r⁻(t, u))."""
        r_net = self._r_plus(t, u) - self._r_minus(t, u)
        return self._S @ r_net

    def jacobian(self, t: float, u: Tensor) -> Tensor:
        """Return ∂f/∂u as an n × n Tensor.

        Uses the analytical Jacobian callable if one was supplied at
        construction, otherwise forward finite differences.
        """
        if self._jac_fn is not None:
            return self._jac_fn(t, u)
        n = u.shape[0]
        backend = u.backend
        f0 = self(t, u)
        cols: list[Tensor] = []
        for j in range(n):
            ej = Tensor.zeros(n, backend=backend)
            ej = ej.set(j, Tensor(self._eps, backend=backend))
            cols.append((self(t, u + ej) - f0) * (1.0 / self._eps))
        rows = [[float(cols[j][i]) for j in range(n)] for i in range(n)]
        return Tensor(rows, backend=backend)


__all__ = ["ReactionNetworkRHS", "project_conserved"]
