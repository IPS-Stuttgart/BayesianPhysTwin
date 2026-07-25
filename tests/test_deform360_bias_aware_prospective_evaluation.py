import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    prospective_case_records,
)
from bayesian_phystwin.deform360_bias_aware_prospective_evaluation import (
    CALIBRATION_GATE_ARTIFACT_KIND,
    CASE_EVALUATION_ARTIFACT_KIND,
    PRIMARY_METRICS,
    aggregate_bias_aware_target_result,
    fit_bias_aware_calibration_gate,
    score_bias_aware_prospective_arrays,
    validate_bias_aware_calibration_gate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPOSITORY_ROOT
    / "configs"
    / "sota"
    / "deform360_bias_aware_guarded_belief_prospective_v1.json"
)
SOURCE_LOCK = (
    REPOSITORY_ROOT
    / "results"
    / "sota"
    / "deform360_bias_aware_guarded_belief_v4"
    / "prospective_lock.json"
)


def _case_report(
    record: dict[str, object],
    *,
    regret_m: float = -0.001,
    eligible: bool = True,
) -> dict[str, object]:
    baseline = 0.010
    prediction = baseline + regret_m
    intervals = []
    for frame, stop in zip((19, 38, 57), (38, 57, 76), strict=True):
        candidate_available = eligible and frame == 19
        intervals.append(
            {
                "frame": frame,
                "interval_end_exclusive": stop,
                "candidate_available": candidate_available,
                "exact_baseline_fallback": not candidate_available,
                "scores": {},
                "regret_m": {metric: regret_m for metric in PRIMARY_METRICS},
                "worst_primary_regret_m": regret_m,
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": CASE_EVALUATION_ARTIFACT_KIND,
        "protocol_id": "deform360-bias-aware-guarded-belief-prospective-v1",
        "protocol_config_sha256": (
            "b6b19be5eaadf830a77f36cccddd38f5b7a35527ca21f7743d2ef147fceabbce"
        ),
        **record,
        "scores": {
            "prediction": {metric: prediction for metric in PRIMARY_METRICS},
            "selected_raw_baseline": {metric: baseline for metric in PRIMARY_METRICS},
        },
        "intervals": intervals,
        "candidate_update_count": int(eligible),
        "all_rejections_bit_exact_fallback": True,
        "authorization": {
            "prediction_cohort_result_sha256": (
                "calibration-cohort"
                if record["role"] == "calibration"
                else "target-cohort"
            ),
            "calibration_gate_result_sha256": (
                None if record["role"] == "calibration" else "calibration-gate"
            ),
        },
    }
    payload["result_sha256"] = canonical_sha256(payload, digest_key="result_sha256")
    return payload


def test_array_scoring_requires_exact_rejected_fallback() -> None:
    point_count = 32
    frame_zero = np.column_stack(
        (
            np.linspace(0.0, 0.1, point_count),
            np.zeros(point_count),
            np.ones(point_count),
        )
    ).astype(np.float32)
    baseline = np.repeat(frame_zero[None], 76, axis=0)
    target = baseline.copy()
    target[1:, :, 1] += np.linspace(0.0, 0.02, 75)[:, None]
    prediction = target.copy()
    support = np.ones((76, point_count), dtype=bool)
    records = [
        {
            "frame": frame,
            "interval_end_exclusive": stop,
            "candidate_available": True,
        }
        for frame, stop in zip((19, 38, 57), (38, 57, 76), strict=True)
    ]

    result = score_bias_aware_prospective_arrays(
        prediction,
        baseline,
        target,
        support,
        support,
        center_ids=np.arange(16),
        update_records=records,
    )

    assert result["candidate_update_count"] == 3
    assert all(
        result["scores"]["prediction"][metric]
        < result["scores"]["selected_raw_baseline"][metric]
        for metric in PRIMARY_METRICS
    )

    records[0]["candidate_available"] = False
    with pytest.raises(ValueError, match="not exact fallback"):
        score_bias_aware_prospective_arrays(
            prediction,
            baseline,
            target,
            support,
            support,
            center_ids=np.arange(16),
            update_records=records,
        )


def test_calibration_gate_combines_source_and_fresh_object_groups() -> None:
    reports = [
        _case_report(record)
        for record in prospective_case_records(PROTOCOL, role="calibration")
    ]
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))

    gate = fit_bias_aware_calibration_gate(
        reports,
        protocol_path=PROTOCOL,
        source_lock=source_lock,
        calibration_cohort_result_sha256="calibration-cohort",
    )

    assert gate["artifact_kind"] == CALIBRATION_GATE_ARTIFACT_KIND
    assert gate["new_eligible_object_group_count"] == 9
    assert gate["combined_eligible_object_group_count"] == 13
    assert gate["finite_sample_rank"] == 13
    assert gate["finite_sample_coverage"] == pytest.approx(13 / 14)
    assert gate["upper_regret_m"] < -gate["minimum_improvement_m"]
    assert gate["calibration_gate_passed"] is True
    assert gate["target_access_authorized"] is True
    validate_bias_aware_calibration_gate(
        gate, protocol_path=PROTOCOL, require_passed=True
    )

    forged = deepcopy(gate)
    forged["upper_regret_m"] = 0.0
    forged["result_sha256"] = canonical_sha256(forged, digest_key="result_sha256")
    with pytest.raises(ValueError, match="finite-sample arithmetic changed"):
        validate_bias_aware_calibration_gate(
            forged, protocol_path=PROTOCOL, require_passed=False
        )


def test_calibration_gate_rejects_one_harmful_accepted_object() -> None:
    records = list(prospective_case_records(PROTOCOL, role="calibration"))
    reports = [_case_report(record) for record in records]
    reports[0] = _case_report(records[0], regret_m=0.001)
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))

    gate = fit_bias_aware_calibration_gate(
        reports,
        protocol_path=PROTOCOL,
        source_lock=source_lock,
        calibration_cohort_result_sha256="calibration-cohort",
    )

    assert gate["accepted_harmful_objects"] == [records[0]["object_id"]]
    assert gate["gates"]["accepted_harmful_object_count"] is False
    assert gate["calibration_gate_passed"] is False
    assert gate["target_access_authorized"] is False
    with pytest.raises(ValueError, match="forbids target access"):
        validate_bias_aware_calibration_gate(
            gate, protocol_path=PROTOCOL, require_passed=True
        )


def test_target_result_uses_object_clusters_and_locked_gates() -> None:
    reports = [
        _case_report(record, regret_m=-0.001)
        for record in prospective_case_records(PROTOCOL, role="target")
    ]

    result = aggregate_bias_aware_target_result(
        reports,
        protocol_path=PROTOCOL,
        target_cohort_result_sha256="target-cohort",
        calibration_gate_result_sha256="calibration-gate",
    )

    assert result["object_count"] == 12
    assert result["episode_count"] == 24
    assert result["object_count_by_stratum"] == {
        "filament": 4,
        "sheet": 4,
        "volumetric": 4,
    }
    assert result["paper_threshold_passed"] is True
    assert all(
        comparison["object_cluster_bootstrap"]["upper_95_difference_m"] < 0.0
        for comparison in result["primary_comparisons"].values()
    )
