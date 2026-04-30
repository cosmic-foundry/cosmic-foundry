"""Parametric reaction-network construction for time-integrator verification.

Builds chain and spoke topologies satisfying the ReactionNetworkRHS protocol
across a range of species counts and rate stiffness ratios.  Detailed balance
(k_f = k_r per pair) guarantees a unique NSE state u_i = 1/n_species for all
i, independent of topology or rate values.

Chain topology
--------------
A₀⇌A₁⇌...⇌A_{n-1}, each pair j: r⁺_j = k_j·A_j, r⁻_j = k_j·A_{j+1}.
Sequential propagation; last pair activates at t ~ n_pairs²/k_min.
Only uniform rates are tested in CI — stiff chains are too slow at Python
speed and are reserved for offline runs once performance improves.

Spoke topology
--------------
Hub A₀ connects to spokes A₁..A_{n-1}, each pair j: r⁺_j = k_j·A₀,
r⁻_j = k_j·A_{j+1}.  All spokes evolve independently; last pair activates
at t ~ 2.3/k_min.  The first half of pairs use rate k_fast; the rest use
k_slow=1.  With k_fast=k_slow=1 the network is uniform.

Generator functions
-------------------
``chain_claims(n_range, k)`` — uniform-rate chain sweep over n_species.
``spoke_claims(n_range, k_ratios)`` — spoke sweep over n_species × k_fast/k_slow.
Both return lists of ``_ParametricNSEClaim`` ready for pytest.mark.parametrize.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators import (
    ConstraintAwareController,
    PIController,
    ReactionNetworkRHS,
    implicit_midpoint,
)
from tests.claims import Claim


@dataclass
class _NetworkSpec:
    """Specification for a parametric reaction network."""

    name: str
    topology: str  # "chain" or "spoke"
    n_species: int
    rates: list[float]  # length n_species − 1; k_f = k_r per pair (detailed balance)

    @property
    def n_pairs(self) -> int:
        return self.n_species - 1

    def t_end(self) -> float:
        if self.topology == "chain":
            return 4.0 * self.n_pairs**2 / min(self.rates)
        return 8.0 / min(self.rates)

    def dt0(self) -> float:
        return min(0.05, 0.1 / max(self.rates))

    def abundance_tol(self) -> float:
        return 1e-7

    def u0(self) -> Tensor:
        return Tensor([1.0] + [0.0] * self.n_pairs)

    def build_rhs(self) -> ReactionNetworkRHS:
        n = self.n_species
        p = self.n_pairs
        rates = list(self.rates)
        topo = self.topology

        s_rows = [[0.0] * p for _ in range(n)]
        if topo == "chain":
            for j in range(p):
                s_rows[j][j] = -1.0
                s_rows[j + 1][j] = 1.0
        else:
            for j in range(p):
                s_rows[0][j] = -1.0
                s_rows[j + 1][j] = 1.0
        S = Tensor(s_rows)
        u0 = self.u0()

        if topo == "chain":

            def r_plus(t: float, u: Tensor) -> Tensor:
                return Tensor(
                    [rates[j] * float(u[j]) for j in range(p)], backend=u.backend
                )

            def r_minus(t: float, u: Tensor) -> Tensor:
                return Tensor(
                    [rates[j] * float(u[j + 1]) for j in range(p)], backend=u.backend
                )

        else:

            def r_plus(t: float, u: Tensor) -> Tensor:
                a0 = float(u[0])
                return Tensor([rates[j] * a0 for j in range(p)], backend=u.backend)

            def r_minus(t: float, u: Tensor) -> Tensor:
                return Tensor(
                    [rates[j] * float(u[j + 1]) for j in range(p)], backend=u.backend
                )

        return ReactionNetworkRHS(S, r_plus, r_minus, u0)


class _ParametricNSEClaim(Claim):
    """NSE detection and equilibrium abundance check for a parametric network.

    Advances the network with ConstraintAwareController to t_end, then
    asserts: (1) NSE was detected, (2) all pairs are active, (3) every
    abundance is within tolerance of 1/n_species.
    """

    def __init__(self, spec: _NetworkSpec) -> None:
        self._spec = spec

    @property
    def description(self) -> str:
        return f"parametric_network/{self._spec.name}"

    def check(self) -> None:
        spec = self._spec
        rhs = spec.build_rhs()
        u0 = spec.u0()
        ctrl = ConstraintAwareController(
            rhs=rhs,
            integrator=implicit_midpoint,
            inner=PIController(alpha=0.35, beta=0.2, tol=1e-5, dt0=spec.dt0()),
            eps_activate=0.01,
            eps_deactivate=0.1,
        )
        state = ctrl.advance(u0, 0.0, spec.t_end())
        assert ctrl.nse_events, f"no NSE events recorded for {spec.name}"
        assert state.active_constraints == frozenset(range(spec.n_pairs)), (
            f"{spec.name}: expected all {spec.n_pairs} pairs active, "
            f"got {state.active_constraints}"
        )
        eq = 1.0 / spec.n_species
        tol = spec.abundance_tol()
        for i in range(spec.n_species):
            xi = float(state.u[i])
            assert abs(xi - eq) < tol, (
                f"{spec.name} species {i}: "
                f"|u[{i}] - {eq:.6g}| = {abs(xi - eq):.3e} >= {tol:.1e}"
            )


def chain_claims(
    n_range: Iterable[int],
    k: float = 1.0,
) -> list[_ParametricNSEClaim]:
    """Generate chain-topology NSE claims for each n_species in n_range.

    All pairs use uniform rate k (forward = reverse, detailed balance).
    Stiff chain variants are slow at Python speed and not included here.
    """
    return [
        _ParametricNSEClaim(
            _NetworkSpec(
                name=f"chain-n{n}-k{k:.0f}",
                topology="chain",
                n_species=n,
                rates=[k] * (n - 1),
            )
        )
        for n in n_range
    ]


def spoke_claims(
    n_range: Iterable[int],
    k_ratios: Iterable[float],
) -> list[_ParametricNSEClaim]:
    """Generate spoke-topology NSE claims for each (n_species, k_ratio) pair.

    The first half of pairs use rate k_ratio; the remaining half use rate 1.0.
    When k_ratio=1 the network is uniform.  The stiffness ratio k_ratio/1
    tests the constraint-aware integrator under mixed rate scales.
    """
    result: list[_ParametricNSEClaim] = []
    for n in n_range:
        p = n - 1
        n_fast = p // 2
        n_slow = p - n_fast
        for k in k_ratios:
            rates = [float(k)] * n_fast + [1.0] * n_slow
            result.append(
                _ParametricNSEClaim(
                    _NetworkSpec(
                        name=f"spoke-n{n}-k{k:.0f}",
                        topology="spoke",
                        n_species=n,
                        rates=rates,
                    )
                )
            )
    return result


__all__ = ["_NetworkSpec", "_ParametricNSEClaim", "chain_claims", "spoke_claims"]
