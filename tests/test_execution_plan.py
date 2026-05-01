"""Execution-plan correctness claims."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.claims import Claim, DeviceCalibration, ExecutionPlan

_CPU_BACKEND = object()
_GPU_BACKEND = object()


def _plan(
    *,
    budget_s: float = 1.0,
    device_kind: str = "cpu",
    cpu_rate: float = 100.0,
    gpu_rate: float | None = None,
) -> ExecutionPlan:
    calibration = DeviceCalibration(
        cpu_backend=_CPU_BACKEND,
        gpu_backend=_GPU_BACKEND if gpu_rate is not None else None,
        cpu_fma_rate=cpu_rate,
        gpu_fma_rate=gpu_rate,
    )
    return ExecutionPlan(
        backend=_GPU_BACKEND if device_kind == "gpu" else _CPU_BACKEND,
        device_kind=device_kind,
        claim_walltime_budget_s=budget_s,
        device_calibration=calibration,
    )


class _ExecutionPlanClaim(Claim[None]):
    def __init__(self, description: str, check: Callable[[], None]) -> None:
        self._description = description
        self._check = check

    @property
    def description(self) -> str:
        return self._description

    def check(self, _calibration: None) -> None:
        self._check()


def _check_batch_size_uses_budget_safety_and_roofline() -> None:
    plan = _plan(budget_s=2.0, cpu_rate=100.0)

    assert (
        plan.batch_size_for(fmas_per_case=10.0, min_batch=4, max_batch=64, safety=0.5)
        == 10
    )


def _check_batch_size_clamps_to_max_when_budget_is_large() -> None:
    plan = _plan(budget_s=10.0, cpu_rate=100.0)

    assert (
        plan.batch_size_for(fmas_per_case=10.0, min_batch=4, max_batch=16, safety=1.0)
        == 16
    )


def _check_problem_size_uses_monotone_work_model() -> None:
    plan = _plan(budget_s=1.0, cpu_rate=1_000.0)

    size = plan.problem_size_for(
        work_fmas=lambda n: float(n**2),
        min_size=4,
        max_size=64,
        safety=0.25,
    )

    assert size == 15


def _check_refinement_count_uses_cumulative_work_model() -> None:
    plan = _plan(budget_s=1.0, cpu_rate=100.0)

    count = plan.refinement_count_for(
        work_fmas=lambda refinements: sum(10.0 * 2**i for i in range(refinements)),
        min_refinements=2,
        max_refinements=8,
        safety=1.0,
    )

    assert count == 3


def _check_extent_skips_when_smallest_debug_extent_exceeds_budget() -> None:
    plan = _plan(budget_s=1.0, cpu_rate=10.0)

    with pytest.raises(pytest.skip.Exception, match="smallest extent 4"):
        plan.batch_size_for(fmas_per_case=10.0, min_batch=4, max_batch=8, safety=0.5)


def _check_gpu_plan_uses_gpu_roofline() -> None:
    plan = _plan(budget_s=1.0, device_kind="gpu", cpu_rate=10.0, gpu_rate=1_000.0)

    assert plan.fma_rate == 1_000.0
    assert (
        plan.batch_size_for(fmas_per_case=100.0, min_batch=1, max_batch=32, safety=1.0)
        == 10
    )


def _check_extent_rejects_min_extent_below_one() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="min_extent"):
        plan.extent_for(
            work_fmas=lambda n: float(n),
            min_extent=0,
            max_extent=4,
            label="bad_extent",
        )


def _check_extent_rejects_max_extent_below_min_extent() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="max_extent"):
        plan.extent_for(
            work_fmas=lambda n: float(n),
            min_extent=4,
            max_extent=3,
            label="bad_extent",
        )


def _check_extent_rejects_nonpositive_safety() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="safety"):
        plan.extent_for(
            work_fmas=lambda n: float(n),
            min_extent=1,
            max_extent=4,
            label="bad_extent",
            safety=0.0,
        )


def _check_extent_rejects_nonpositive_work_estimate() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="work estimate"):
        plan.problem_size_for(work_fmas=lambda n: 0.0, min_size=1, max_size=4)


_CORRECTNESS_CLAIMS: list[Claim[None]] = [
    _ExecutionPlanClaim(
        "extent/batch_budget_safety_roofline",
        _check_batch_size_uses_budget_safety_and_roofline,
    ),
    _ExecutionPlanClaim(
        "extent/batch_clamps_to_max",
        _check_batch_size_clamps_to_max_when_budget_is_large,
    ),
    _ExecutionPlanClaim(
        "extent/problem_size_monotone_work",
        _check_problem_size_uses_monotone_work_model,
    ),
    _ExecutionPlanClaim(
        "extent/refinement_count_cumulative_work",
        _check_refinement_count_uses_cumulative_work_model,
    ),
    _ExecutionPlanClaim(
        "extent/skip_smallest_debug_extent",
        _check_extent_skips_when_smallest_debug_extent_exceeds_budget,
    ),
    _ExecutionPlanClaim("extent/gpu_roofline", _check_gpu_plan_uses_gpu_roofline),
    _ExecutionPlanClaim(
        "extent/reject_min_extent",
        _check_extent_rejects_min_extent_below_one,
    ),
    _ExecutionPlanClaim(
        "extent/reject_max_extent",
        _check_extent_rejects_max_extent_below_min_extent,
    ),
    _ExecutionPlanClaim(
        "extent/reject_safety",
        _check_extent_rejects_nonpositive_safety,
    ),
    _ExecutionPlanClaim(
        "extent/reject_work_estimate",
        _check_extent_rejects_nonpositive_work_estimate,
    ),
]


@pytest.mark.parametrize(
    "claim", _CORRECTNESS_CLAIMS, ids=[c.description for c in _CORRECTNESS_CLAIMS]
)
def test_correctness(claim: Claim[None]) -> None:
    claim.check(None)
