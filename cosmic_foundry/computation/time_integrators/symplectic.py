"""Symplectic splitting integrators for separable Hamiltonian systems.

A separable Hamiltonian H(q, p) = T(p) + V(q) splits Hamilton's equations into
two exactly-integrable flows:

    Drift (kinetic):  dq/dt = ∂T/∂p,  dp/dt = 0
    Kick  (potential): dq/dt = 0,       dp/dt = −∂V/∂q

A splitting method with coefficient vectors c and d alternates s drift/kick
pairs per step:

    for i = 0, …, s−1:
        q ← q + c[i]·dt·(∂T/∂p)(p)        (drift)
        p ← p + d[i]·dt·(−∂V/∂q)(q)       (kick)

Higher-order methods come from the Suzuki-Yoshida triple-jump composition:
applying a base method at scales (α₁, α₀, α₁) where α₁ = 1/(2−2^e) and
α₀ = 1−2α₁, with e = 1/(2k+1) to increase order from 2k to 2k+2.

Named instances (all ABA-form, d[-1] = 0):
    symplectic_euler — order 1 (non-palindromic, for comparison)
    leapfrog         — order 2 (Störmer-Verlet)
    forest_ruth      — order 4
    yoshida_6        — order 6
    yoshida_8        — order 8
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from cosmic_foundry.computation.tensor import Tensor


@runtime_checkable
class HamiltonianSplitProtocol(Protocol):
    """Protocol for separable Hamiltonians H(q, p) = T(p) + V(q).

    Implementers supply the two gradient maps needed by the drift/kick split:
    the velocity ∂T/∂p and the force −∂V/∂q.
    """

    def dT_dp(self, p: Tensor) -> Tensor:
        """Velocity q̇ = ∂T/∂p evaluated at momentum p."""
        ...

    def dV_dq(self, q: Tensor) -> Tensor:
        """Potential gradient ∂V/∂q evaluated at position q."""
        ...


class HamiltonianSplit:
    """Concrete HamiltonianSplitProtocol wrapping two gradient callables.

    Parameters
    ----------
    dT_dp:
        Callable (p: Tensor) → Tensor returning the velocity ∂T/∂p.
    dV_dq:
        Callable (q: Tensor) → Tensor returning the potential gradient ∂V/∂q.
        The kick applies −dV_dq to p, so a harmonic potential V = q²/2 gives
        dV_dq = q and the kick is p ← p − d·dt·q.
    """

    def __init__(
        self,
        dT_dp: Callable[[Tensor], Tensor],
        dV_dq: Callable[[Tensor], Tensor],
    ) -> None:
        self._dT_dp = dT_dp
        self._dV_dq = dV_dq

    def dT_dp(self, p: Tensor) -> Tensor:
        result: Tensor = self._dT_dp(p)
        return result

    def dV_dq(self, q: Tensor) -> Tensor:
        result: Tensor = self._dV_dq(q)
        return result


class PartitionedState:
    """Integrator state for symplectic methods.

    Carries the current time and the canonical coordinates (position q,
    momentum p) of the Hamiltonian system.

    Parameters
    ----------
    t:
        Current time.
    q:
        Position (generalized coordinates) as a Tensor.
    p:
        Momentum (conjugate to q) as a Tensor.
    """

    __slots__ = ("t", "q", "p")

    def __init__(self, t: float, q: Tensor, p: Tensor) -> None:
        self.t = t
        self.q = q
        self.p = p


class SymplecticSplittingIntegrator:
    """Explicit symplectic integrator parameterized by drift/kick coefficients.

    Advances a partitioned Hamiltonian system (q, p) by one step of size dt
    using alternating drift and kick sub-steps with weights c[i] and d[i].

    Parameters
    ----------
    c:
        Drift weights, length s.  Must sum to 1.
    d:
        Kick weights, length s.  Must sum to 1.  ABA-type methods set d[-1] = 0.
    order:
        Declared convergence order.
    """

    def __init__(self, c: list[float], d: list[float], order: int) -> None:
        self._c = list(c)
        self._d = list(d)
        self._order = order
        self._s = len(c)

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._order

    def step(
        self,
        H: HamiltonianSplitProtocol,
        state: PartitionedState,
        dt: float,
    ) -> PartitionedState:
        """Advance state by one step of size dt.

        Returns a new PartitionedState with t = state.t + dt and updated
        (q, p) from the alternating drift/kick sequence.
        """
        q, p = state.q, state.p
        for ci, di in zip(self._c, self._d, strict=True):
            if ci != 0.0:
                q = q + (ci * dt) * H.dT_dp(p)
            if di != 0.0:
                p = p - (di * dt) * H.dV_dq(q)
        return PartitionedState(state.t + dt, q, p)


# ---------------------------------------------------------------------------
# Coefficient construction via Suzuki-Yoshida triple-jump composition
# ---------------------------------------------------------------------------


def _triple_jump(
    c: list[float], d: list[float], exponent: float
) -> tuple[list[float], list[float]]:
    """Lift the order of an ABA splitting method by 2 via triple-jump composition.

    Given a method with ABA coefficients (c, d) of order 2k, returns the
    coefficients for the order-(2k+2) method obtained by composing three
    applications at scales (α₁, α₀, α₁) with α₁ = 1/(2 − 2^exponent)
    and α₀ = 1 − 2α₁.  Adjacent drift half-steps at the seams are merged.

    The exponent for lifting from order 2k to 2k+2 is e = 1/(2k+1).
    """
    a1 = 1.0 / (2.0 - 2.0**exponent)
    a0 = 1.0 - 2.0 * a1
    c1 = [a1 * x for x in c]
    c0 = [a0 * x for x in c]
    d1 = [a1 * x for x in d]
    d0 = [a0 * x for x in d]
    # Merge the trailing half-drift of each sub-method with the leading
    # half-drift of the next (the d[-1]=0 null kick is absorbed implicitly).
    c_new = c1[:-1] + [c1[-1] + c0[0]] + c0[1:-1] + [c0[-1] + c1[0]] + c1[1:]
    d_new = d1[:-1] + d0[:-1] + d1[:-1] + [0.0]
    return c_new, d_new


# ---------------------------------------------------------------------------
# Named instances
# ---------------------------------------------------------------------------

symplectic_euler = SymplecticSplittingIntegrator(
    c=[1.0],
    d=[1.0],
    order=1,
)

_c_lf = [0.5, 0.5]
_d_lf = [1.0, 0.0]
leapfrog = SymplecticSplittingIntegrator(c=_c_lf, d=_d_lf, order=2)

_c_fr, _d_fr = _triple_jump(_c_lf, _d_lf, exponent=1 / 3)
forest_ruth = SymplecticSplittingIntegrator(c=_c_fr, d=_d_fr, order=4)

_c_y6, _d_y6 = _triple_jump(_c_fr, _d_fr, exponent=1 / 5)
yoshida_6 = SymplecticSplittingIntegrator(c=_c_y6, d=_d_y6, order=6)

_c_y8, _d_y8 = _triple_jump(_c_y6, _d_y6, exponent=1 / 7)
yoshida_8 = SymplecticSplittingIntegrator(c=_c_y8, d=_d_y8, order=8)


__all__ = [
    "HamiltonianSplit",
    "HamiltonianSplitProtocol",
    "PartitionedState",
    "SymplecticSplittingIntegrator",
    "forest_ruth",
    "leapfrog",
    "symplectic_euler",
    "yoshida_6",
    "yoshida_8",
]
