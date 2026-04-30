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

The state is an ODEState where u = concat([q, p]) and HamiltonianRHSProtocol
carries split_index to indicate how many leading components are position (q);
the remaining components are momentum (p).  This makes SymplecticCompositionIntegrator
a TimeIntegrator that interoperates with Integrator.

Supported orders: 1 (symplectic Euler), 2 (leapfrog/Störmer-Verlet), 4
(Forest-Ruth), 6 (Yoshida), 8 (Yoshida).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators.integrator import (
    ODEState,
    TimeIntegrator,
)


@runtime_checkable
class HamiltonianRHSProtocol(Protocol):
    """Protocol for separable Hamiltonians H(q, p) = T(p) + V(q).

    Implementers supply the two gradient maps needed by the drift/kick split
    and the split_index that partitions the concatenated state vector u into
    position q = u[:split_index] and momentum p = u[split_index:].
    """

    @property
    def split_index(self) -> int:
        """Number of position (q) components; momentum starts at u[split_index:]."""
        ...

    def dT_dp(self, p: Tensor) -> Tensor:
        """Velocity q̇ = ∂T/∂p evaluated at momentum p."""
        ...

    def dV_dq(self, q: Tensor) -> Tensor:
        """Potential gradient ∂V/∂q evaluated at position q."""
        ...


class HamiltonianRHS:
    """Concrete HamiltonianRHSProtocol wrapping two gradient callables.

    Parameters
    ----------
    dT_dp:
        Callable (p: Tensor) → Tensor returning the velocity ∂T/∂p.
    dV_dq:
        Callable (q: Tensor) → Tensor returning the potential gradient ∂V/∂q.
        The kick applies −dV_dq to p, so a harmonic potential V = q²/2 gives
        dV_dq = q and the kick is p ← p − d·dt·q.
    split_index:
        Number of position components in the concatenated state vector u.
        Positions occupy u[:split_index]; momenta occupy u[split_index:].
    """

    def __init__(
        self,
        dT_dp: Callable[[Tensor], Tensor],
        dV_dq: Callable[[Tensor], Tensor],
        split_index: int,
    ) -> None:
        self._dT_dp = dT_dp
        self._dV_dq = dV_dq
        self._split_index = split_index

    @property
    def split_index(self) -> int:
        """Number of position (q) components."""
        return self._split_index

    def dT_dp(self, p: Tensor) -> Tensor:
        result: Tensor = self._dT_dp(p)
        return result

    def dV_dq(self, q: Tensor) -> Tensor:
        result: Tensor = self._dV_dq(q)
        return result


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


def _build_symplectic_coefficients() -> dict[int, tuple[list[float], list[float]]]:
    c2, d2 = [0.5, 0.5], [1.0, 0.0]
    c4, d4 = _triple_jump(c2, d2, exponent=1 / 3)
    c6, d6 = _triple_jump(c4, d4, exponent=1 / 5)
    c8, d8 = _triple_jump(c6, d6, exponent=1 / 7)
    return {
        1: ([1.0], [1.0]),
        2: (c2, d2),
        4: (c4, d4),
        6: (c6, d6),
        8: (c8, d8),
    }


_SYMPLECTIC_COEFFICIENTS: dict[int, tuple[list[float], list[float]]] = (
    _build_symplectic_coefficients()
)


class SymplecticCompositionIntegrator(TimeIntegrator):
    """Explicit symplectic integrator for separable Hamiltonian systems.

    Advances the state one step by alternating drift and kick sub-steps with
    weights c[i] and d[i] derived from the Suzuki-Yoshida triple-jump
    composition of order-2 leapfrog.  All coefficient computation is internal.

    The state is an ODEState where u = concat([q, p]).  The HamiltonianRHSProtocol
    carries split_index so that q = u[:split_index] and p = u[split_index:].
    This makes the integrator fully compatible with Integrator.advance().

    Supported orders and their methods:
        1 — symplectic Euler (non-palindromic, for comparison)
        2 — leapfrog / Störmer-Verlet
        4 — Forest-Ruth
        6 — Yoshida order 6
        8 — Yoshida order 8

    Parameters
    ----------
    order:
        Convergence order.  Must be one of {1, 2, 4, 6, 8}.
    """

    def __init__(self, order: int) -> None:
        if order not in _SYMPLECTIC_COEFFICIENTS:
            raise ValueError(
                f"SymplecticCompositionIntegrator order must be one of "
                f"{sorted(_SYMPLECTIC_COEFFICIENTS)}, got {order}"
            )
        c, d = _SYMPLECTIC_COEFFICIENTS[order]
        self._c = list(c)
        self._d = list(d)
        self._order = order

    @property
    def order(self) -> int:
        """Declared convergence order."""
        return self._order

    def step(
        self,
        H: HamiltonianRHSProtocol,
        state: ODEState,
        dt: float,
    ) -> ODEState:
        """Advance state by one step of size dt.

        Unpacks q = state.u[:split_index] and p = state.u[split_index:],
        applies the alternating drift/kick sequence, then returns a new ODEState
        with u = concat([q_new, p_new]) and t = state.t + dt.
        """
        n = H.split_index
        q = state.u[:n]
        p = state.u[n:]
        for ci, di in zip(self._c, self._d, strict=True):
            if ci != 0.0:
                q = q + (ci * dt) * H.dT_dp(p)
            if di != 0.0:
                p = p - (di * dt) * H.dV_dq(q)
        q_list: list[float] = q.to_list()
        p_list: list[float] = p.to_list()
        u_new: Tensor = Tensor(q_list + p_list, backend=state.u.backend)
        return ODEState(state.t + dt, u_new)


__all__ = [
    "HamiltonianRHS",
    "HamiltonianRHSProtocol",
    "SymplecticCompositionIntegrator",
]
