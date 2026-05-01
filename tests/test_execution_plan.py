"""Execution-plan correctness claims."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.claims import (
    BATCH_REPLAY_INDEX_ENV,
    DEVICE_GPU_CPU_MIN_SPEEDUP,
    BatchedFailure,
    Claim,
    DeviceCalibration,
    ExecutionPlan,
    gpu_trust_skip_reason,
)

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


def _check_gpu_trust_accepts_material_speedup() -> None:
    assert (
        gpu_trust_skip_reason(
            cpu_rate=100.0, gpu_rate=DEVICE_GPU_CPU_MIN_SPEEDUP * 100.0
        )
        is None
    )


def _check_gpu_trust_rejects_missing_rate() -> None:
    reason = gpu_trust_skip_reason(100.0, None)

    assert reason is not None
    assert "did not produce a rate" in reason


def _check_gpu_trust_rejects_invalid_rate() -> None:
    reason = gpu_trust_skip_reason(100.0, 0.0)

    assert reason is not None
    assert "invalid rate" in reason


def _check_gpu_trust_rejects_cpu_speed_device() -> None:
    reason = gpu_trust_skip_reason(
        cpu_rate=100.0, gpu_rate=(DEVICE_GPU_CPU_MIN_SPEEDUP * 100.0) - 1.0
    )

    assert reason is not None
    assert "requires" in reason


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


def _check_batch_indices_default_to_full_batch() -> None:
    plan = _plan()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.delenv(BATCH_REPLAY_INDEX_ENV, raising=False)
        assert plan.batch_indices_for(4, label="claim") == (0, 1, 2, 3)


def _check_batch_indices_select_replay_lane() -> None:
    plan = _plan()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(BATCH_REPLAY_INDEX_ENV, "2")
        assert plan.batch_indices_for(4, label="claim") == (2,)


def _check_batch_indices_reject_noninteger_replay_lane() -> None:
    plan = _plan()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(BATCH_REPLAY_INDEX_ENV, "two")
        with pytest.raises(ValueError, match=BATCH_REPLAY_INDEX_ENV):
            plan.batch_indices_for(4, label="claim")


def _check_batch_indices_reject_out_of_range_replay_lane() -> None:
    plan = _plan()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(BATCH_REPLAY_INDEX_ENV, "4")
        with pytest.raises(ValueError, match="outside batch index range"):
            plan.batch_indices_for(4, label="claim")


def _check_batched_failure_message_contains_replay_metadata() -> None:
    message = BatchedFailure(
        claim="claim",
        device_kind="cpu",
        batch_size=8,
        batch_index=3,
        method="method",
        order=4,
        problem="problem",
        parameters={"rate": 1.5},
        actual=0.1,
        expected=0.2,
        error=0.1,
        tolerance=0.01,
    ).format()

    assert "batch_index=3" in message
    assert "method=method" in message
    assert "order=4" in message
    assert "problem=problem" in message
    assert "rate=1.5" in message
    assert f"{BATCH_REPLAY_INDEX_ENV}=3" in message


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
        "calibration/gpu_trust_accepts_speedup",
        _check_gpu_trust_accepts_material_speedup,
    ),
    _ExecutionPlanClaim(
        "calibration/gpu_trust_rejects_missing_rate",
        _check_gpu_trust_rejects_missing_rate,
    ),
    _ExecutionPlanClaim(
        "calibration/gpu_trust_rejects_invalid_rate",
        _check_gpu_trust_rejects_invalid_rate,
    ),
    _ExecutionPlanClaim(
        "calibration/gpu_trust_rejects_cpu_speed_device",
        _check_gpu_trust_rejects_cpu_speed_device,
    ),
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
    _ExecutionPlanClaim(
        "replay/full_batch_without_override",
        _check_batch_indices_default_to_full_batch,
    ),
    _ExecutionPlanClaim(
        "replay/select_lane",
        _check_batch_indices_select_replay_lane,
    ),
    _ExecutionPlanClaim(
        "replay/reject_noninteger_lane",
        _check_batch_indices_reject_noninteger_replay_lane,
    ),
    _ExecutionPlanClaim(
        "replay/reject_out_of_range_lane",
        _check_batch_indices_reject_out_of_range_replay_lane,
    ),
    _ExecutionPlanClaim(
        "replay/failure_metadata",
        _check_batched_failure_message_contains_replay_metadata,
    ),
]


@pytest.mark.parametrize(
    "claim", _CORRECTNESS_CLAIMS, ids=[c.description for c in _CORRECTNESS_CLAIMS]
)
def test_correctness(claim: Claim[None]) -> None:
    claim.check(None)
