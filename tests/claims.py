"""Shared base classes, calibration types, and budget constants for test claims."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import sympy

from cosmic_foundry.computation.tensor import Tensor
from cosmic_foundry.theory.discrete.discrete_field import _CallableDiscreteField

C = TypeVar("C")

# ── Solver convergence budget ─────────────────────────────────────────────────
# Controls mesh-refinement N_max sizing for test_convergence.py claims only.
# Set via CF_SOLVER_CONVERGENCE_BUDGET_S (default 5 s locally, 60 s in CI).
SOLVER_CONVERGENCE_BUDGET_S: float = float(
    os.environ.get("CF_SOLVER_CONVERGENCE_BUDGET_S", "5.0")
)

# ── Time-integrator test budgets ──────────────────────────────────────────────
# INTEGRATOR_CLAIM_BUDGET_S  : per-claim wall-time cap for the halving loop in
#                test_time_integrators.py.  Governs how many dt halvings are
#                attempted before moving on.  Set via
#                CF_INTEGRATOR_CLAIM_BUDGET_S (default 1 s locally and in CI).
# INTEGRATOR_SESSION_BUDGET_S : expected total wall time for the entire
#                test_time_integrators.py session (convergence claims +
#                NSE claims + behavior checks).  Feeds the session-level
#                timeout alongside SOLVER_CONVERGENCE_BUDGET_S so neither
#                suite's budget inflates the other's.  Set via
#                CF_INTEGRATOR_SESSION_BUDGET_S (default 30 s locally, 90 s in CI).
INTEGRATOR_CLAIM_BUDGET_S: float = float(
    os.environ.get("CF_INTEGRATOR_CLAIM_BUDGET_S", "1.0")
)
INTEGRATOR_SESSION_BUDGET_S: float = float(
    os.environ.get("CF_INTEGRATOR_SESSION_BUDGET_S", "30.0")
)

# ── Shared fixed overhead ─────────────────────────────────────────────────────
# Per-session overhead not covered by either convergence budget: performance-
# test calibration, structure/tensor tests, solver calibration (one probe per
# solver type per session), and GPU benchmark variability.
FIXED_SESSION_OVERHEAD_S: float = 40.0

# Tolerance multiplier on the total expected session time.
BUDGET_TOLERANCE: float = 1.1


class Claim(ABC):
    """Base for static correctness claims that do not depend on calibration."""

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self) -> None: ...


class CalibratedClaim(ABC, Generic[C]):
    """Base for claims whose verification requires runtime calibration data.

    The type parameter C is the calibration type.  Bind it concretely in each
    claim family, e.g. CalibratedClaim[float] for FMA-rate-calibrated claims.
    """

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def check(self, calibration: C) -> None: ...


@dataclass(frozen=True)
class DeviceCalibration:
    """FMA throughput rooflines and backend instances for available compute devices.

    cpu_backend and cpu_fma_rate always refer to the CPU device.
    gpu_backend and gpu_fma_rate are None when no functional GPU backend is
    available (no device found, or XLA/driver error during measurement).

    The backends stored here are the exact instances used during calibration;
    performance claims should use them for benchmarking so that the measured
    roofline and the claim workload run through the same code paths.
    """

    cpu_backend: Any
    gpu_backend: Any | None
    cpu_fma_rate: float
    gpu_fma_rate: float | None


def assemble_linear_op(disc: Any, mesh: Any) -> Any:
    """Return a pre-assembled sparse object satisfying the LinearOperator protocol.

    Probes disc symbolically on mesh to extract the sparse stiffness structure
    (rows, cols, vals).  The returned object provides apply(), diagonal(), and
    row_abs_sums() — the three methods required by LinearSolver.solve().

    The mesh is used only during this call; it is not retained by the returned
    object.  disc must be a linear DiscreteOperator: the probe assumes that each
    output expression is linear in the symbolic input values.
    """
    n = mesh.n_cells
    shape = mesh.shape

    def _to_flat(idx: tuple[int, ...]) -> int:
        flat, stride = 0, 1
        for a, i in enumerate(idx):
            flat += i * stride
            stride *= shape[a]
        return flat

    def _to_multi(flat: int) -> tuple[int, ...]:
        idx = []
        for s in shape:
            idx.append(flat % s)
            flat //= s
        return tuple(idx)

    u_syms = [sympy.Symbol(f"_u{j}") for j in range(n)]
    sym_field = _CallableDiscreteField(mesh, lambda idx: u_syms[_to_flat(idx)])
    result = disc(sym_field)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        expr = result(_to_multi(i))
        for j, sym in enumerate(u_syms):
            c = float(expr.coeff(sym))
            if c != 0.0:
                rows.append(i)
                cols.append(j)
                vals.append(c)

    class _Assembled:
        def apply(self, u: Tensor) -> Tensor:
            backend = u.backend
            raw = backend.spmv(rows, cols, vals, u._value, n)
            return Tensor(raw, backend=backend)

        def diagonal(self, backend: Any) -> Tensor:
            d = [0.0] * n
            for r, c, v in zip(rows, cols, vals, strict=True):
                if r == c:
                    d[r] += v
            return Tensor(d, backend=backend)

        def row_abs_sums(self, backend: Any) -> Tensor:
            s = [0.0] * n
            for r, v in zip(rows, vals, strict=True):
                s[r] += abs(v)
            return Tensor(s, backend=backend)

    return _Assembled()
