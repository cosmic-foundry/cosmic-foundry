"""Shared ODE calculation fixtures for time-integrator tests."""

from __future__ import annotations

import math
from typing import Any

import cosmic_foundry.computation.time_integrators as _ti
from cosmic_foundry.computation.backends import NumpyBackend
from cosmic_foundry.computation.tensor import Tensor

TIME_BACKEND = NumpyBackend()


def exact2(t: float) -> tuple[float, ...]:
    e = math.exp(-t)
    return e, 1.0 - e


def exact3(t: float) -> tuple[float, ...]:
    e1, e2 = math.exp(-t), math.exp(-2.0 * t)
    return e1, e1 - e2, 1.0 - 2.0 * e1 + e2


def exact_osc(t: float) -> tuple[float, ...]:
    return math.cos(t), math.sin(t)


def exact_ham(t: float) -> tuple[float, ...]:
    return math.cos(t), -math.sin(t)


def exact_scalar_decay(t: float) -> tuple[float, ...]:
    return (math.exp(-t),)


def exact_semilinear(t: float) -> tuple[float, ...]:
    forced = (
        math.exp(-2.0 * t)
        * (math.exp(2.0 * t) * (2.0 * math.sin(t) - math.cos(t)) + 1.0)
        / 5.0
    )
    return (math.exp(-2.0 * t) + forced,)


def err(u: Tensor, exact: Any, t: float) -> float:
    return max(abs(float(u[i]) - v) for i, v in enumerate(exact(t)))


def conserved(u: Tensor, n: int) -> bool:
    return abs(sum(float(u[i]) for i in range(n)) - 1.0) < 1e-10


def scalar_decay_jacobian_rhs() -> _ti.JacobianRHS:
    return _ti.JacobianRHS(
        lambda t, u: Tensor([-float(u[0])], backend=u.backend),
        lambda t, u: Tensor([[-1.0]], backend=u.backend),
    )


def split_decay_rhs() -> _ti.SplitRHS:
    return _ti.SplitRHS(
        lambda t, u: Tensor([-0.2 * float(u[0])], backend=u.backend),
        lambda t, u: Tensor([-0.8 * float(u[0])], backend=u.backend),
        lambda t, u: Tensor([[-0.8]], backend=u.backend),
    )


def semilinear_forcing_rhs(backend: Any = TIME_BACKEND) -> _ti.SemilinearRHS:
    return _ti.SemilinearRHS(
        Tensor([[-2.0]], backend=backend),
        lambda t, u: Tensor([math.sin(t)], backend=u.backend),
    )


def harmonic_hamiltonian_rhs() -> _ti.HamiltonianRHS:
    return _ti.HamiltonianRHS(
        dT_dp=lambda p: p,
        dV_dq=lambda q: q,
        split_index=1,
    )


def oscillator_composite_rhs() -> _ti.CompositeRHS:
    return _ti.CompositeRHS(
        [
            _ti.ComponentFlowRHS(
                lambda t, u: Tensor([-float(u[1]), 0.0], backend=u.backend),
                symplectic_form_defect_upper_bound=0.0,
            ),
            _ti.ComponentFlowRHS(
                lambda t, u: Tensor([0.0, float(u[0])], backend=u.backend),
                symplectic_form_defect_upper_bound=0.0,
            ),
        ]
    )
