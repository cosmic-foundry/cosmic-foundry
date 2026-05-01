"""Execution-plan extent selection tests."""

from __future__ import annotations

import pytest

from tests.claims import DeviceCalibration, ExecutionPlan

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


def test_batch_size_uses_budget_safety_and_roofline() -> None:
    plan = _plan(budget_s=2.0, cpu_rate=100.0)

    assert (
        plan.batch_size_for(fmas_per_case=10.0, min_batch=4, max_batch=64, safety=0.5)
        == 10
    )


def test_batch_size_clamps_to_max_when_budget_is_large() -> None:
    plan = _plan(budget_s=10.0, cpu_rate=100.0)

    assert (
        plan.batch_size_for(fmas_per_case=10.0, min_batch=4, max_batch=16, safety=1.0)
        == 16
    )


def test_problem_size_uses_monotone_work_model() -> None:
    plan = _plan(budget_s=1.0, cpu_rate=1_000.0)

    size = plan.problem_size_for(
        work_fmas=lambda n: float(n**2),
        min_size=4,
        max_size=64,
        safety=0.25,
    )

    assert size == 15


def test_refinement_count_uses_cumulative_work_model() -> None:
    plan = _plan(budget_s=1.0, cpu_rate=100.0)

    count = plan.refinement_count_for(
        work_fmas=lambda refinements: sum(10.0 * 2**i for i in range(refinements)),
        min_refinements=2,
        max_refinements=8,
        safety=1.0,
    )

    assert count == 3


def test_extent_skips_when_smallest_debug_extent_exceeds_budget() -> None:
    plan = _plan(budget_s=1.0, cpu_rate=10.0)

    with pytest.raises(pytest.skip.Exception, match="smallest extent 4"):
        plan.batch_size_for(fmas_per_case=10.0, min_batch=4, max_batch=8, safety=0.5)


def test_gpu_plan_uses_gpu_roofline() -> None:
    plan = _plan(budget_s=1.0, device_kind="gpu", cpu_rate=10.0, gpu_rate=1_000.0)

    assert plan.fma_rate == 1_000.0
    assert (
        plan.batch_size_for(fmas_per_case=100.0, min_batch=1, max_batch=32, safety=1.0)
        == 10
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_extent": 0, "max_extent": 4}, "min_extent"),
        ({"min_extent": 4, "max_extent": 3}, "max_extent"),
        ({"min_extent": 1, "max_extent": 4, "safety": 0.0}, "safety"),
    ],
)
def test_extent_rejects_invalid_bounds(kwargs: dict[str, float], match: str) -> None:
    plan = _plan()

    with pytest.raises(ValueError, match=match):
        plan.extent_for(
            work_fmas=lambda n: float(n),
            label="bad_extent",
            **kwargs,
        )


def test_extent_rejects_nonpositive_work_estimate() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="work estimate"):
        plan.problem_size_for(work_fmas=lambda n: 0.0, min_size=1, max_size=4)
