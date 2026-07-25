from __future__ import annotations

import copy
import json
from pathlib import Path

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
)
from bayesian_phystwin.deform360_bias_aware_prospective_evaluation import (
    CASE_EVALUATION_ARTIFACT_KIND,
    PRIMARY_METRICS,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    PROTOCOL_ID as V1_PROTOCOL_ID,
    load_bias_aware_prospective_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_calibration import (
    fit_v2_calibration_accuracy_gate,
    validate_v2_calibration_access,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_protocol import (
    PROTOCOL_ID as V2_PROTOCOL_ID,
    load_bias_aware_prospective_v2_protocol,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_runtime import (
    prospective_v2_case_records,
)
from bayesian_phystwin.deform360_bias_aware_prospective_v2_support import (
    COHORT_ARTIFACT_KIND,
    build_v2_calibration_support_gate,
)


ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL = (
    ROOT / "configs/sota/deform360_bias_aware_guarded_belief_prospective_v1.json"
)
V2_PROTOCOL = (
    ROOT / "configs/sota/deform360_bias_aware_guarded_belief_prospective_v2.json"
)
SOURCE_LOCK = (
    ROOT / "results/sota/deform360_bias_aware_guarded_belief_v4/prospective_lock.json"
)
AUTOMATIC = {
    "076-rubber-bands",
    "011-green-cloth",
    "175-plastic-bag-cloth",
    "163-bear",
    "168-cat-big",
    "078-fishing-line",
    "161-tube",
    "088-snake",
}
FRESH = {"078-fishing-line", "161-tube", "088-snake"}


def _write_support_artifacts(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    protocol = load_bias_aware_prospective_v2_protocol(V2_PROTOCOL)
    rows = []
    for record in prospective_v2_case_records(V2_PROTOCOL, role="calibration"):
        automatic = record["object_id"] in AUTOMATIC
        rows.append(
            {
                **record,
                "origin": (
                    "fresh_v2" if record["object_id"] in FRESH else "inherited_v1"
                ),
                "disposition": "prediction" if automatic else "quality_failure",
                "physical_mode": "warp_twin" if automatic else None,
                "automatic_twin": automatic,
                "eligible_for_accuracy_and_calibration": automatic,
                "prediction_seal_result_sha256": (
                    f"prediction-{record['case']}" if automatic else None
                ),
            }
        )
    cohort: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": COHORT_ARTIFACT_KIND,
        "protocol_id": V2_PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        "role": "calibration",
        "expected_case_count": 12,
        "prediction_count": 8,
        "quality_failure_count": 4,
        "automatic_twin_count": 8,
        "replacement_count": 0,
        "cases": rows,
        "complete": True,
        "information_boundary": {
            "base_dispositions_inherited_without_relabeling": True,
            "fresh_predictions_or_failures_sealed_before_future_open": True,
            "calibration_future_read": False,
            "calibration_outcome_read": False,
            "target_media_read": False,
            "target_future_read": False,
            "replacement_allowed": False,
        },
    }
    cohort["result_sha256"] = canonical_sha256(cohort, digest_key="result_sha256")
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    support_path = tmp_path / "support.json"
    support = build_v2_calibration_support_gate(
        V2_PROTOCOL,
        cohort_seal_path=cohort_path,
        output_path=support_path,
    )
    return cohort_path, support_path, support


def _report(record: dict[str, object], *, regret_m: float) -> dict[str, object]:
    origin = "fresh_v2" if record["object_id"] in FRESH else "inherited_v1"
    protocol = (
        load_bias_aware_prospective_v2_protocol(V2_PROTOCOL)
        if origin == "fresh_v2"
        else load_bias_aware_prospective_protocol(V1_PROTOCOL)
    )
    baseline = 0.01
    prediction = baseline + regret_m
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": CASE_EVALUATION_ARTIFACT_KIND,
        "protocol_id": V2_PROTOCOL_ID if origin == "fresh_v2" else V1_PROTOCOL_ID,
        "protocol_config_sha256": protocol["config_sha256"],
        **record,
        "scores": {
            "prediction": {metric: prediction for metric in PRIMARY_METRICS},
            "selected_raw_baseline": {metric: baseline for metric in PRIMARY_METRICS},
        },
        "intervals": [
            {
                "candidate_available": True,
                "worst_primary_regret_m": regret_m,
            }
        ],
        "all_rejections_bit_exact_fallback": True,
    }
    report["result_sha256"] = canonical_sha256(report, digest_key="result_sha256")
    return report


def test_support_gate_authorizes_only_automatic_calibration_cases(
    tmp_path: Path,
) -> None:
    cohort, support, gate = _write_support_artifacts(tmp_path)

    record, disposition, observed = validate_v2_calibration_access(
        V2_PROTOCOL,
        cohort_seal_path=cohort,
        support_gate_path=support,
        object_id="088-snake",
        episode_id=1,
        expected_origin="fresh_v2",
    )
    assert record["case"] == "088-snake-ep0001"
    assert disposition["automatic_twin"] is True
    assert observed["result_sha256"] == gate["result_sha256"]

    try:
        validate_v2_calibration_access(
            V2_PROTOCOL,
            cohort_seal_path=cohort,
            support_gate_path=support,
            object_id="160-hose",
            episode_id=1,
        )
    except ValueError as error:
        assert "eligible automatic twin" in str(error)
    else:
        raise AssertionError("quality failure was authorized")


def test_v2_accuracy_gate_passes_uniform_safe_improvement(tmp_path: Path) -> None:
    cohort_path, support_path, _ = _write_support_artifacts(tmp_path)
    records = [
        dict(record)
        for record in prospective_v2_case_records(V2_PROTOCOL, role="calibration")
        if record["object_id"] in AUTOMATIC
    ]
    reports = [_report(record, regret_m=-1.0e-5) for record in records]
    base = load_bias_aware_prospective_protocol(V1_PROTOCOL)
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))

    gate = fit_v2_calibration_accuracy_gate(
        reports,
        protocol_path=V2_PROTOCOL,
        base_protocol_config_sha256=base["config_sha256"],
        cohort_seal_path=cohort_path,
        support_gate_path=support_path,
        source_lock=source_lock,
    )

    assert gate["calibration_gate_passed"] is True
    assert gate["target_access_authorized"] is True
    assert gate["new_eligible_object_group_count"] == 8
    assert gate["combined_eligible_object_group_count"] == 12
    assert gate["finite_sample_coverage"] == 12 / 13


def test_v2_accuracy_gate_rejects_one_harmful_accepted_object(
    tmp_path: Path,
) -> None:
    cohort_path, support_path, _ = _write_support_artifacts(tmp_path)
    records = [
        dict(record)
        for record in prospective_v2_case_records(V2_PROTOCOL, role="calibration")
        if record["object_id"] in AUTOMATIC
    ]
    reports = [_report(record, regret_m=-1.0e-5) for record in records]
    harmful = copy.deepcopy(reports[-1])
    harmful["scores"]["prediction"] = {metric: 0.01002 for metric in PRIMARY_METRICS}
    harmful["intervals"][0]["worst_primary_regret_m"] = 2.0e-5
    harmful["result_sha256"] = canonical_sha256(harmful, digest_key="result_sha256")
    reports[-1] = harmful
    base = load_bias_aware_prospective_protocol(V1_PROTOCOL)
    source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))

    gate = fit_v2_calibration_accuracy_gate(
        reports,
        protocol_path=V2_PROTOCOL,
        base_protocol_config_sha256=base["config_sha256"],
        cohort_seal_path=cohort_path,
        support_gate_path=support_path,
        source_lock=source_lock,
    )

    assert gate["calibration_gate_passed"] is False
    assert gate["target_access_authorized"] is False
    assert gate["gates"]["accepted_harmful_object_count"] is False
