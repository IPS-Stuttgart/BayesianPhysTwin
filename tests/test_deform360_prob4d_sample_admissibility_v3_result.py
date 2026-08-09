from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesian_phystwin.deform360_prob4d_sample_admissibility import (
    validate_deform360_prob4d_sample_admissibility_result,
)
from bayesian_phystwin.deform360_prob4d_sample_admissibility_contract import (
    validate_deform360_prob4d_sample_admissibility_policy,
)

SOURCE = Path("results/sota/deform360_prob4d_sample_admissibility_v3/source-artifact")
SAMPLE_ROOT = SOURCE / "sample-admissibility"
IMPLEMENTATION_REVISION = "0beaadab170e644fbaf3b4241d89d950e7a889ef"
POLICY_ID = "25c0a43b720accb3bacd16933774b3773a6bc951443b02b88498ca542d5fc51c"
RESULT_ID = "55ba8f58217e0720025e2b01ab075dae2cabc41f0209c274872ab08cdf206f19"


def _verify_manifest(root: Path) -> None:
    records = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    for record in records:
        digest, relative = record.split(maxsplit=1)
        path = root / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_frozen_v3_source_artifact_is_integrity_bound() -> None:
    _verify_manifest(SOURCE)
    _verify_manifest(SAMPLE_ROOT)


def test_frozen_v3_admissibility_decision_is_terminal_support_negative() -> None:
    result = validate_deform360_prob4d_sample_admissibility_result(SAMPLE_ROOT)

    assert result["result_id"] == RESULT_ID
    assert result["implementation_revision"] == IMPLEMENTATION_REVISION
    assert result["sample_admissibility_policy_id"] == POLICY_ID
    assert result["status"] == "sample-admissibility-gate-failed"
    assert result["admitted_stream_count"] == 324
    assert result["prior_excluded_stream_count"] == 11
    assert result["candidate_stream_count"] == 313
    assert result["admissible_stream_count"] == 102
    assert result["support_negative_stream_count"] == 211
    assert result["technical_failure_stream_count"] == 0
    assert result["supported_object_count"] == 9
    assert result["plan_emitted"] is False
    assert result["plan_id"] is None
    assert {job["status"] for job in result["jobs"]} == {
        "admissible",
        "support-negative",
    }
    assert {
        job["failure_reason"]
        for job in result["jobs"]
        if job["status"] == "support-negative"
    } == {"insufficient-target-free-held-prefix-sample-support"}


def test_frozen_v3_failure_is_dominated_by_spatial_redundancy() -> None:
    result = validate_deform360_prob4d_sample_admissibility_result(SAMPLE_ROOT)
    failures = {"gauge": 0, "clusters": 0, "held_rows": 0}
    admissible_by_object: dict[str, int] = {}
    for job in result["jobs"]:
        object_id = str(job["object_id"])
        admissible_by_object.setdefault(object_id, 0)
        if job["status"] == "admissible":
            admissible_by_object[object_id] += 1
        windows = job["windows"]
        failures["gauge"] += any(
            window["metric_gauge_correspondence_count"] < 8 for window in windows
        )
        failures["clusters"] += any(
            window["metric_gauge_spatial_cluster_count"] < 8 for window in windows
        )
        failures["held_rows"] += any(
            window["held_prefix_point_row_count"] < 32 for window in windows
        )

    assert failures == {"gauge": 21, "clusters": 211, "held_rows": 14}
    assert sorted(admissible_by_object.values()) == [0, 2, 3, 6, 9, 10, 10, 15, 17, 30]


def test_frozen_v3_pipeline_kept_outcomes_closed() -> None:
    pipeline = json.loads((SOURCE / "pipeline-receipt.json").read_text())
    policy = validate_deform360_prob4d_sample_admissibility_policy(
        json.loads((SAMPLE_ROOT / "sample-admissibility-policy.json").read_text())
    )

    assert policy["artifact_id"] == POLICY_ID
    assert pipeline["implementation_revision"] == IMPLEMENTATION_REVISION
    assert pipeline["sample_admissibility_policy_id"] == POLICY_ID
    assert pipeline["stage_outcomes"] == {
        "calibration": "skipped",
        "metric_batch": "success",
        "sample_admissibility": "success",
        "sample_admissibility_gate": "failure",
        "samples": "skipped",
        "source_gate": "skipped",
        "support_gate": "success",
    }
    assert pipeline["source_gate_passed"] is None
    assert pipeline["source_gate_result_id"] is None
    assert pipeline["confirmation_access_authorized"] is None
    assert pipeline["information_boundary"] == {
        "confirmation_payloads_opened": False,
        "future_frames_used": False,
        "human_approval_required": False,
        "new_measurements_required": False,
        "public_released_measurements_used": True,
        "replacement_allowed": False,
        "target_outcomes_used": False,
    }
    assert all(
        digest == hashlib.sha256(b"").hexdigest()
        for digest in pipeline["stderr_sha256"].values()
    )
