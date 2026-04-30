"""Problem setups, exact solutions, and convergence helpers for time-integrator tests.

All test problems and shared infrastructure used by test_time_integrators.py live here
so the test file can focus on claims and registries.  Nothing in this module asserts;
it only builds objects and computes values.
"""

from __future__ import annotations

import math
import sys

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.computation.time_integrators import (
    BlackBoxRHS,
    CompositeRHS,
    JacobianRHS,
    ODEState,
    SemilinearRHS,
    SplitRHS,
)

# ---------------------------------------------------------------------------
# Shared convergence knobs
# ---------------------------------------------------------------------------

DT_BASE: float = 0.1
N_HALVINGS: int = 5
BDF_N_HALVINGS: int = 7  # extra halvings for BDF/AM: bootstrap pre-asymptotic transient
PI_ALPHA: float = 0.7
PI_BETA: float = 0.4

# ---------------------------------------------------------------------------
# Base 2-species network: X0 → X1 with rate k = 1
#
# f(t, u) = [−k·X0, k·X0]
# Exact from (1, 0): X0(t) = exp(−k·t), X1(t) = 1 − X0(t)
# ---------------------------------------------------------------------------

_BASE_K: float = 1.0


def base_network_f(t: float, u: Tensor) -> Tensor:
    x0 = float(u[0])
    return Tensor([-_BASE_K * x0, _BASE_K * x0], backend=u.backend)


def _base_network_jac(t: float, u: Tensor) -> Tensor:
    return Tensor([[-_BASE_K, 0.0], [_BASE_K, 0.0]], backend=u.backend)


BASE_RHS: JacobianRHS = JacobianRHS(f=base_network_f, jac=_base_network_jac)

BASE_RHS_IMEX: SplitRHS = SplitRHS(
    f_E=lambda t, u: Tensor([0.0, _BASE_K * float(u[0])], backend=u.backend),
    f_I=lambda t, u: Tensor([-_BASE_K * float(u[0]), 0.0], backend=u.backend),
    jac_I=lambda t, u: Tensor([[-_BASE_K, 0.0], [0.0, 0.0]], backend=u.backend),
)


def base_network_exact(t: float) -> tuple[float, float]:
    x0 = math.exp(-_BASE_K * t)
    return x0, 1.0 - x0


def max_base_network_error(u: Tensor, t: float) -> float:
    x0, x1 = base_network_exact(t)
    return max(abs(float(u[0]) - x0), abs(float(u[1]) - x1))


# ---------------------------------------------------------------------------
# 3-species decay chain: A → B → C, rates k1=1, k2=2
#
# Exact from X(0) = [1, 0, 0]:
#   X0(t) = exp(−t),  X1(t) = exp(−t) − exp(−2t),  X2(t) = 1 − X0 − X1
# ---------------------------------------------------------------------------

DECAY_K1: float = 1.0
DECAY_K2: float = 2.0

_DECAY_JAC_VALS: list[list[float]] = [
    [-DECAY_K1, 0.0, 0.0],
    [DECAY_K1, -DECAY_K2, 0.0],
    [0.0, DECAY_K2, 0.0],
]


def decay_f(t: float, u: Tensor) -> Tensor:
    x0, x1 = float(u[0]), float(u[1])
    return Tensor(
        [-DECAY_K1 * x0, DECAY_K1 * x0 - DECAY_K2 * x1, DECAY_K2 * x1],
        backend=u.backend,
    )


def _decay_jac(t: float, u: Tensor) -> Tensor:
    return Tensor(_DECAY_JAC_VALS, backend=u.backend)


DECAY_RHS: JacobianRHS = JacobianRHS(f=decay_f, jac=_decay_jac)

DECAY_RHS_IMEX: SplitRHS = SplitRHS(
    f_E=lambda t, u: Tensor(
        [0.0, DECAY_K1 * float(u[0]), DECAY_K2 * float(u[1])], backend=u.backend
    ),
    f_I=lambda t, u: Tensor(
        [-DECAY_K1 * float(u[0]), -DECAY_K2 * float(u[1]), 0.0], backend=u.backend
    ),
    jac_I=lambda t, u: Tensor(
        [[-DECAY_K1, 0.0, 0.0], [0.0, -DECAY_K2, 0.0], [0.0, 0.0, 0.0]],
        backend=u.backend,
    ),
)


def decay_exact(t: float) -> tuple[float, float, float]:
    x0 = math.exp(-DECAY_K1 * t)
    x1 = math.exp(-DECAY_K1 * t) - math.exp(-DECAY_K2 * t)
    return x0, x1, 1.0 - x0 - x1


def max_decay_error(u: Tensor, t: float) -> float:
    x0, x1, x2 = decay_exact(t)
    return max(abs(float(u[i]) - v) for i, v in enumerate((x0, x1, x2)))


# ---------------------------------------------------------------------------
# Variable-regime 3-species problems used by variable-order and stiffness tests
# ---------------------------------------------------------------------------


def sharpening_decay_f(t: float, u: Tensor) -> Tensor:
    """Decay chain whose second rate jumps from 1 to 10 at t = 0.5."""
    k2 = 1.0 if t < 0.5 else 10.0
    x0, x1 = float(u[0]), float(u[1])
    return Tensor([-x0, x0 - k2 * x1, k2 * x1], backend=u.backend)


def stiffening_decay_f(t: float, u: Tensor) -> Tensor:
    """Decay chain whose second rate jumps from 1 to 80 at t = 0.5."""
    k2 = 1.0 if t < 0.5 else 80.0
    x0, x1 = float(u[0]), float(u[1])
    return Tensor([-x0, x0 - k2 * x1, k2 * x1], backend=u.backend)


def stiffening_decay_jac(t: float, u: Tensor) -> Tensor:
    k2 = 1.0 if t < 0.5 else 80.0
    return Tensor([[-1.0, 0.0, 0.0], [1.0, -k2, 0.0], [0.0, k2, 0.0]])


STIFFENING_RHS: JacobianRHS = JacobianRHS(
    f=stiffening_decay_f, jac=stiffening_decay_jac
)


def vode_fast_slow_f(t: float, u: Tensor) -> Tensor:
    """Decay chain whose second rate jumps from 1 to 1000 at t = 0.45."""
    k2 = 1.0 if t < 0.45 else 1000.0
    x0, x1 = float(u[0]), float(u[1])
    return Tensor([-x0, x0 - k2 * x1, k2 * x1], backend=u.backend)


def vode_fast_slow_jac(t: float, u: Tensor) -> Tensor:
    k2 = 1.0 if t < 0.45 else 1000.0
    return Tensor([[-1.0, 0.0, 0.0], [1.0, -k2, 0.0], [0.0, k2, 0.0]])


VODE_RHS: JacobianRHS = JacobianRHS(f=vode_fast_slow_f, jac=vode_fast_slow_jac)

# ---------------------------------------------------------------------------
# ETD semilinear split: stiff linear part + slowly-varying nonlinear residual
# ---------------------------------------------------------------------------


def etd_split_rhs() -> SemilinearRHS:
    """Mass-conserving semilinear RHS for ETD convergence and mass-fraction tests.

    Linear part: capture chain X0 → X1 → X2 with rates 8, 0.5.
    Nonlinear residual: slow additional path X0 → X2 with rate 0.25 + 0.1·sin(2t).
    Column sums of the combined Jacobian are zero, so mass fractions are conserved.
    """
    linear = Tensor([[-8.0, 0.0, 0.0], [8.0, -0.5, 0.0], [0.0, 0.5, 0.0]])

    def residual(t: float, u: Tensor) -> Tensor:
        x0 = float(u[0])
        rate = 0.25 + 0.1 * math.sin(2.0 * t)
        return Tensor([-rate * x0, 0.0, rate * x0], backend=u.backend)

    return SemilinearRHS(linear, residual)


def integrate_etd(inst: object, dt: float, t_end: float = 0.5) -> ODEState:
    rhs = etd_split_rhs()
    state = ODEState(0.0, Tensor([1.0, 0.0, 0.0]))
    for _ in range(round(t_end / dt)):
        state = inst.step(rhs, state, dt)  # type: ignore[union-attr]
    return state


# ---------------------------------------------------------------------------
# Operator-splitting 2D oscillator
#
# du/dt = (A + B)u with A = [[0, −w], [0, 0]], B = [[0, 0], [w, 0]], w = 1.
# Exact: rotation by angle w·t.  [A, B] ≠ 0, so splitting error reveals order.
# ---------------------------------------------------------------------------

_SPLIT_OMEGA: float = 1.0


def _split_a(t: float, u: Tensor) -> Tensor:
    return Tensor([-_SPLIT_OMEGA * float(u[1]), 0.0], backend=u.backend)


def _split_b(t: float, u: Tensor) -> Tensor:
    return Tensor([0.0, _SPLIT_OMEGA * float(u[0])], backend=u.backend)


def split_rhs() -> CompositeRHS:
    return CompositeRHS([BlackBoxRHS(_split_a), BlackBoxRHS(_split_b)])


def split_exact(t: float, u0: Tensor) -> Tensor:
    c, s = math.cos(_SPLIT_OMEGA * t), math.sin(_SPLIT_OMEGA * t)
    return Tensor(
        [c * float(u0[0]) - s * float(u0[1]), s * float(u0[0]) + c * float(u0[1])],
        backend=u0.backend,
    )


def integrate_split(inst: object, dt: float, t_end: float = 1.0) -> ODEState:
    rhs = split_rhs()
    state = ODEState(0.0, Tensor([1.0, 0.0]))
    for _ in range(round(t_end / dt)):
        state = inst.step(rhs, state, dt)  # type: ignore[union-attr]
    return state


# ---------------------------------------------------------------------------
# Convergence slope measurement
# ---------------------------------------------------------------------------


def measure_convergence_slope(
    errors: list[float],
    dts: list[float],
    label: str = "",
) -> float:
    """Compute log-log convergence slope; excludes machine-precision points.

    Raises AssertionError if fewer than 3 points are above floating-point noise.
    The caller is responsible for asserting slope ≥ declared_order − tolerance.
    """
    eps = sys.float_info.epsilon * 10
    valid = [(dt, e) for dt, e in zip(dts, errors, strict=False) if e > eps]
    if len(valid) < 3:
        raise AssertionError(
            f"{label + ': ' if label else ''}error reached machine precision too early "
            f"({len(valid)} valid points above {eps:.1e})"
        )
    log_dts = [math.log(dt) for dt, _ in valid]
    log_errs = [math.log(e) for _, e in valid]
    n = len(log_dts)
    mean_x = sum(log_dts) / n
    mean_y = sum(log_errs) / n
    return sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(log_dts, log_errs, strict=False)
    ) / sum((x - mean_x) ** 2 for x in log_dts)


# ---------------------------------------------------------------------------
# Integration run helpers (used by _ConvergenceClaim and _ConservationClaim)
# ---------------------------------------------------------------------------


def run_rk(inst: object, rhs: object, u0: Tensor, dt: float) -> ODEState:
    """Step inst for ceil(1.0/dt) steps from ODEState(0, u0)."""
    state = ODEState(0.0, u0)
    for _ in range(math.ceil(1.0 / dt)):
        state = inst.step(rhs, state, dt)  # type: ignore[union-attr]
    return state


def run_multistep(inst: object, rhs: object, u0: Tensor, dt: float) -> ODEState:
    """Bootstrap inst with init_state then step to t ≈ 1.

    Uses ceil(1.0/dt) − inst.order post-bootstrap steps so total elapsed
    time is approximately 1.0 regardless of order.
    """
    order: int = inst.order  # type: ignore[union-attr]
    n_steps = max(math.ceil(1.0 / dt) - order, 1)
    state = inst.init_state(rhs, 0.0, u0, dt)  # type: ignore[union-attr]
    for _ in range(n_steps):
        state = inst.step(rhs, state, dt)  # type: ignore[union-attr]
    return state


def run_conservation(
    inst: object,
    rhs: object,
    u0: Tensor,
    dt: float,
    t_end: float,
) -> ODEState:
    """Step inst for round(t_end/dt) steps from ODEState(0, u0)."""
    state = ODEState(0.0, u0)
    for _ in range(round(t_end / dt)):
        state = inst.step(rhs, state, dt)  # type: ignore[union-attr]
    return state


def run_multistep_conservation(
    inst: object,
    rhs: object,
    u0: Tensor,
    dt: float,
    t_end: float,
) -> ODEState:
    """Bootstrap inst with init_state then run to t_end."""
    order: int = inst.order  # type: ignore[union-attr]
    n_steps = max(round(t_end / dt) - order, 1)
    state = inst.init_state(rhs, 0.0, u0, dt)  # type: ignore[union-attr]
    for _ in range(n_steps):
        state = inst.step(rhs, state, dt)  # type: ignore[union-attr]
    return state


# ---------------------------------------------------------------------------
# Conservation / positivity checker
# ---------------------------------------------------------------------------


def assert_conservation(
    u: Tensor,
    n: int,
    *,
    label: str,
    conservation_tol: float = 1e-12,
) -> None:
    """Assert Σuᵢ = 1 and uᵢ ≥ −1e-10 for an n-component mass-fraction state."""
    total = sum(float(u[i]) for i in range(n))
    assert abs(total - 1.0) < conservation_tol, (
        f"{label}: sum(X) = {total:.15f} ≠ 1 "
        f"(mass not conserved to {conservation_tol:.1e})"
    )
    for i in range(n):
        xi = float(u[i])
        assert xi >= -1e-10, f"{label}: X[{i}] = {xi:.3e} < 0"


__all__ = [
    "DT_BASE",
    "N_HALVINGS",
    "BDF_N_HALVINGS",
    "PI_ALPHA",
    "PI_BETA",
    "BASE_RHS",
    "BASE_RHS_IMEX",
    "DECAY_RHS",
    "DECAY_RHS_IMEX",
    "STIFFENING_RHS",
    "VODE_RHS",
    "base_network_f",
    "base_network_exact",
    "max_base_network_error",
    "decay_f",
    "decay_exact",
    "max_decay_error",
    "sharpening_decay_f",
    "etd_split_rhs",
    "integrate_etd",
    "split_rhs",
    "split_exact",
    "integrate_split",
    "measure_convergence_slope",
    "run_rk",
    "run_multistep",
    "run_conservation",
    "run_multistep_conservation",
    "assert_conservation",
]
