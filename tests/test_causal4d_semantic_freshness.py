import json
from pathlib import Path

import numpy as np
import pytest

import causal4d
from causal4d.contracts import TaskPosterior, build_causal_context
from causal4d.semantic_freshness import (
    SEMANTIC_TIMING_SCHEMA_VERSION,
    SEMANTIC_TIMING_SCOPE,
    SemanticFreshnessDecision,
    SemanticFreshnessLimits,
    SemanticTimingMetadata,
    apply_semantic_freshness_gate,
)


def _task() -> TaskPosterior:
    observations = np.zeros((6, 1, 3), dtype=float)
    actions = np.zeros((6, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="semantic_freshness_unit",
        case_id="case",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    return TaskPosterior(
        context=context,
        physical_posterior_id="1" * 64,
        component_ids=("left", "right"),
        physical_weights=np.asarray([0.6, 0.4]),
        task_weights=np.asarray([0.8, 0.2]),
        semantic_log_scores=np.asarray([0.5, -0.5]),
        beta=2.0,
        query_node_indices=np.asarray([0]),
        semantic_source="unit-semantic-model",
        metadata={"evidence_id": "kept-on-fallback"},
    )


def _timing() -> dict[str, object]:
    return {
        "schema_version": 1,
        "clock": "monotonic",
        "clock_id": "planner-process-17",
        "timing_scope": SEMANTIC_TIMING_SCOPE,
        "query_created_monotonic_s": 100.0,
        "generation_started_monotonic_s": 100.1,
        "generation_completed_monotonic_s": 100.5,
        "current_monotonic_s": 100.6,
        "planning_deadline_monotonic_s": 101.0,
    }


def _limits() -> dict[str, object]:
    return {
        "schema_version": 1,
        "max_inference_duration_s": 1.0,
        "max_forecast_age_s": 2.0,
    }


def _gate(task, timing, limits, *, expected_clock_id="planner-process-17"):
    return apply_semantic_freshness_gate(
        task,
        timing,
        limits,
        expected_clock_id=expected_clock_id,
    )


def _assert_physical_only(original: TaskPosterior, gated: TaskPosterior) -> None:
    assert gated.beta == 0.0
    assert gated.physical_weights.tobytes() == original.physical_weights.tobytes()
    assert gated.task_weights.tobytes() == original.physical_weights.tobytes()
    assert gated.physical_weights.tobytes() == gated.task_weights.tobytes()
    assert gated.component_ids == original.component_ids
    assert np.array_equal(gated.semantic_log_scores, original.semantic_log_scores)
    assert gated.semantic_source == original.semantic_source
    assert gated.metadata == original.metadata


def test_freshness_api_is_exported_from_package() -> None:
    assert causal4d.SEMANTIC_TIMING_SCHEMA_VERSION == SEMANTIC_TIMING_SCHEMA_VERSION
    assert causal4d.SEMANTIC_TIMING_SCOPE == SEMANTIC_TIMING_SCOPE
    assert causal4d.SemanticFreshnessDecision is SemanticFreshnessDecision
    assert causal4d.SemanticFreshnessLimits is SemanticFreshnessLimits
    assert causal4d.SemanticTimingMetadata is SemanticTimingMetadata
    assert causal4d.apply_semantic_freshness_gate is apply_semantic_freshness_gate


def test_fresh_semantic_evidence_is_preserved_exactly() -> None:
    task = _task()
    gated, decision = _gate(task, _timing(), _limits())

    assert gated is task
    assert decision.accepted
    assert decision.applied_beta == task.beta
    assert decision.reasons == ()
    assert decision.inference_duration_s == pytest.approx(0.4)
    assert decision.forecast_age_s == pytest.approx(0.6)
    assert decision.planning_deadline_margin_s == pytest.approx(0.4)
    assert decision.clock_id == "planner-process-17"
    assert decision.expected_clock_id == "planner-process-17"
    assert decision.timing_scope == SEMANTIC_TIMING_SCOPE


def test_missing_or_mismatched_trusted_clock_id_fails_closed() -> None:
    task = _task()

    missing_task, missing = apply_semantic_freshness_gate(task, _timing(), _limits())
    assert missing.reasons == ("invalid_expected_clock_id",)
    _assert_physical_only(task, missing_task)

    mismatch_task, mismatch = _gate(
        task,
        _timing(),
        _limits(),
        expected_clock_id="planner-process-18",
    )
    assert mismatch.reasons == ("clock_id_mismatch",)
    assert mismatch.clock_id == "planner-process-17"
    assert mismatch.expected_clock_id == "planner-process-18"
    _assert_physical_only(task, mismatch_task)


@pytest.mark.parametrize(
    ("updates", "expected_reason"),
    [
        (
            {
                "generation_completed_monotonic_s": 101.2,
                "current_monotonic_s": 101.3,
                "planning_deadline_monotonic_s": 103.0,
            },
            "semantic_inference_timeout",
        ),
        (
            {
                "current_monotonic_s": 102.1,
                "planning_deadline_monotonic_s": 103.0,
            },
            "stale_semantic_forecast",
        ),
        (
            {"planning_deadline_monotonic_s": 100.55},
            "planning_deadline_missed",
        ),
    ],
)
def test_late_or_stale_semantic_evidence_falls_back(
    updates: dict[str, object], expected_reason: str
) -> None:
    task = _task()
    timing = {**_timing(), **updates}
    gated, decision = _gate(task, timing, _limits())

    assert not decision.accepted
    assert expected_reason in decision.reasons
    assert decision.applied_beta == 0.0
    _assert_physical_only(task, gated)


def test_limit_and_deadline_boundaries_are_inclusive() -> None:
    task = _task()
    timing = {
        **_timing(),
        "generation_started_monotonic_s": 100.0,
        "generation_completed_monotonic_s": 101.0,
        "current_monotonic_s": 102.0,
        "planning_deadline_monotonic_s": 102.0,
    }
    gated, decision = _gate(task, timing, _limits())

    assert gated is task
    assert decision.accepted


@pytest.mark.parametrize(
    "mutation",
    [
        lambda timing: timing.pop("current_monotonic_s"),
        lambda timing: timing.pop("clock_id"),
        lambda timing: timing.update(extra_field=1),
        lambda timing: timing.update(clock="wall_clock"),
        lambda timing: timing.update(clock_id=""),
        lambda timing: timing.update(timing_scope="model_kernel_only"),
        lambda timing: timing.update(query_created_monotonic_s=-1.0),
        lambda timing: timing.update(current_monotonic_s=float("nan")),
        lambda timing: timing.update(current_monotonic_s="100.6"),
        lambda timing: timing.update(current_monotonic_s=True),
        lambda timing: timing.update(generation_started_monotonic_s=99.9),
        lambda timing: timing.update(generation_completed_monotonic_s=100.0),
        lambda timing: timing.update(current_monotonic_s=100.4),
        lambda timing: timing.update(planning_deadline_monotonic_s=99.9),
    ],
)
def test_malformed_negative_or_inconsistent_timing_fails_closed(mutation) -> None:
    task = _task()
    timing = _timing()
    mutation(timing)
    gated, decision = _gate(task, timing, _limits())

    assert not decision.accepted
    assert decision.reasons == ("invalid_timing_metadata",)
    assert decision.validation_error
    _assert_physical_only(task, gated)


@pytest.mark.parametrize(
    "limits",
    [
        {"schema_version": 1, "max_inference_duration_s": 1.0},
        {
            "schema_version": 1,
            "max_inference_duration_s": -1.0,
            "max_forecast_age_s": 2.0,
        },
        {
            "schema_version": 1,
            "max_inference_duration_s": 1.0,
            "max_forecast_age_s": float("inf"),
        },
    ],
)
def test_invalid_limits_fail_closed(limits: dict[str, object]) -> None:
    task = _task()
    gated, decision = _gate(task, _timing(), limits)

    assert decision.reasons == ("invalid_freshness_limits",)
    _assert_physical_only(task, gated)


def test_typed_timing_and_limits_are_supported() -> None:
    task = _task()
    timing = SemanticTimingMetadata.from_mapping(_timing())
    limits = SemanticFreshnessLimits.from_mapping(_limits())
    gated, decision = _gate(task, timing, limits)

    assert gated is task
    assert decision.accepted


def test_hardware_gate_requires_bounded_fail_closed_semantics() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "causal4d"
        / "hardware_execution_gate_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    freshness = payload["semantic_branch"]["freshness_gate"]

    assert freshness["required_for_positive_beta"] is True
    assert freshness["clock"] == "monotonic"
    assert freshness["shared_clock_id_required"] is True
    assert (
        freshness["expected_clock_id_source"]
        == "trusted_planner_runtime_context_not_semantic_timing_payload"
    )
    assert freshness["reject_if_reported_clock_id_mismatches_expected"] is True
    assert freshness["generation_timing_scope"] == SEMANTIC_TIMING_SCOPE
    assert freshness["runtime_timing_schema_version"] == 1
    limits = SemanticFreshnessLimits.from_mapping(freshness["limits"])
    assert limits.max_inference_duration_s > 0.0
    assert limits.max_forecast_age_s > 0.0
    assert freshness["planning_deadline_required_per_query"] is True
    assert freshness["reject_if_current_time_exceeds_planning_deadline"] is True
    assert freshness["malformed_timing_fails_closed"] is True
    assert freshness["fallback"] == {
        "beta": 0.0,
        "preserve_physical_weights_bit_for_bit": True,
        "semantic_branch_must_not_block_hardware": True,
    }
