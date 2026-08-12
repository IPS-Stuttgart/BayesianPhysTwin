from __future__ import annotations

import copy
import hashlib

import pytest

import bayesian_phystwin.deform360_fresh_object_session_candidate_runner_v6_1 as runner
from bayesian_phystwin.deform360_fresh_object_session_candidate_runner_v6_1 import (
    build_deform360_v61_candidate_execution_receipt,
    build_deform360_v61_candidate_panel_receipt,
    build_deform360_v61_candidate_technical_failure_receipt,
    validate_deform360_v61_candidate_execution_receipt,
    validate_deform360_v61_candidate_panel_receipt,
    validate_deform360_v61_candidate_technical_failure_receipt,
)
from bayesian_phystwin.deform360_fresh_object_session_candidate_v6_1 import (
    CANDIDATE_AMENDMENT_FILE_SHA256,
    EXECUTION_LOCK_FILE_SHA256,
    UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
    UPSTREAM_EXECUTION_RECEIPT_ID,
    UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
    UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
    UPSTREAM_SOURCE_PLAN_FILE_SHA256,
)
from bayesian_phystwin.deform360_fresh_object_session_source_v6_1 import (
    UPSTREAM_PREDICTION_BATCH_ID,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _mapping(label: str) -> dict[str, str]:
    return {
        f"{outer:02d}-{target:02d}": _digest(f"{label}/{outer}/{target}")
        for outer in range(10)
        for target in range(10)
    }


def _batch() -> dict[str, object]:
    return {
        "prediction_batch_id": _digest("raw-batch"),
        "record_count": 100,
    }


def _receipt() -> dict[str, object]:
    return build_deform360_v61_candidate_panel_receipt(
        candidate_revision="2" * 40,
        upstream_prediction_receipt_id=_digest("upstream-receipt"),
        raw_prediction_batch=_batch(),
        raw_prediction_batch_file_sha256=_digest("raw-batch-file"),
        candidate_artifact_id_by_record=_mapping("artifact"),
        candidate_seal_file_sha256_by_record=_mapping("seal"),
        raw_record_file_sha256_by_record=_mapping("record"),
        technical_failure_record_count=3,
    )


def test_candidate_panel_receipt_is_complete_and_closed() -> None:
    receipt = validate_deform360_v61_candidate_panel_receipt(
        _receipt(),
        raw_prediction_batch=_batch(),
        raw_prediction_batch_file_sha256=_digest("raw-batch-file"),
    )

    assert receipt["upstream_prediction_batch_id"] == UPSTREAM_PREDICTION_BATCH_ID
    assert receipt["prediction_record_count"] == 100
    assert receipt["technical_failure_record_count"] == 3
    assert receipt["information_boundary"]["source_suffix_opened"] is False
    assert receipt["information_boundary"]["source_suffix_access_authorized"] is False
    assert (
        receipt["information_boundary"]["independent_confirmation_authorized"] is False
    )
    assert runner.SEALED_VISUAL_PRODUCT_FILENAME == "baseline_disjoint.npz"


def test_candidate_panel_receipt_rejects_incomplete_roster() -> None:
    artifacts = _mapping("artifact")
    artifacts.pop("09-09")

    with pytest.raises(ValueError, match="roster changed"):
        build_deform360_v61_candidate_panel_receipt(
            candidate_revision="2" * 40,
            upstream_prediction_receipt_id=_digest("upstream-receipt"),
            raw_prediction_batch=_batch(),
            raw_prediction_batch_file_sha256=_digest("raw-batch-file"),
            candidate_artifact_id_by_record=artifacts,
            candidate_seal_file_sha256_by_record=_mapping("seal"),
            raw_record_file_sha256_by_record=_mapping("record"),
            technical_failure_record_count=0,
        )


def test_candidate_panel_receipt_rejects_identity_and_boundary_drift() -> None:
    changed = copy.deepcopy(_receipt())
    changed["information_boundary"]["source_suffix_opened"] = True

    with pytest.raises(ValueError, match="contract changed"):
        validate_deform360_v61_candidate_panel_receipt(changed)

    changed = copy.deepcopy(_receipt())
    changed["receipt_id"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        validate_deform360_v61_candidate_panel_receipt(changed)


def test_candidate_panel_receipt_rejects_wrong_raw_batch() -> None:
    changed_batch = _batch()
    changed_batch["prediction_batch_id"] = _digest("other-batch")

    with pytest.raises(ValueError, match="another raw batch"):
        validate_deform360_v61_candidate_panel_receipt(
            _receipt(), raw_prediction_batch=changed_batch
        )


def _execution_artifacts(receipt: dict[str, object]) -> dict[str, str]:
    return {
        "candidate_amendment": CANDIDATE_AMENDMENT_FILE_SHA256,
        "candidate_panel_receipt": _digest("candidate-panel-receipt-file"),
        "candidate_raw_batch": str(receipt["raw_prediction_batch_file_sha256"]),
        "execution_lock": EXECUTION_LOCK_FILE_SHA256,
        "upstream_execution_receipt": UPSTREAM_EXECUTION_RECEIPT_FILE_SHA256,
        "upstream_prediction_batch": UPSTREAM_PREDICTION_BATCH_FILE_SHA256,
        "upstream_prediction_receipt": UPSTREAM_PREDICTION_RECEIPT_FILE_SHA256,
        "upstream_source_plan": UPSTREAM_SOURCE_PLAN_FILE_SHA256,
    }


def test_candidate_execution_receipt_binds_one_closed_protected_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _receipt()
    monkeypatch.setattr(
        runner,
        "_validate_upstream_execution_receipt",
        lambda _value: {"receipt_id": UPSTREAM_EXECUTION_RECEIPT_ID},
    )
    receipt = build_deform360_v61_candidate_execution_receipt(
        candidate_revision="2" * 40,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        upstream_execution_receipt={},
        candidate_panel_receipt=panel,
        artifact_file_sha256=_execution_artifacts(panel),
    )

    validated = validate_deform360_v61_candidate_execution_receipt(receipt)
    assert validated["prediction_record_count"] == 100
    assert validated["source_suffix_access_authorized"] is False
    assert validated["independent_confirmation_authorized"] is False
    boundary = validated["information_boundary"]
    assert boundary["prob4d_pipeline_artifacts_reused"] is True
    assert boundary["prob4d_decoded_uniform_fusion_used"] is False
    assert boundary["motioncrafter_disjoint_baseline_used"] is True

    changed = copy.deepcopy(validated)
    changed["source_suffix_access_authorized"] = True
    with pytest.raises(ValueError, match="identity changed"):
        validate_deform360_v61_candidate_execution_receipt(changed)


def test_candidate_technical_failure_is_terminal_and_closed() -> None:
    receipt = build_deform360_v61_candidate_technical_failure_receipt(
        candidate_revision="2" * 40,
        runner_name="workstation2",
        workflow_run_id=123,
        workflow_run_attempt=1,
        terminal_stage="publish-candidate-panel",
        exit_code=7,
        retained_artifact_file_sha256={"logs/runner.log": _digest("log")},
    )

    validated = validate_deform360_v61_candidate_technical_failure_receipt(receipt)
    assert validated["exit_code"] == 7
    assert validated["status"] == "candidate-prefix-technical-failure-retained"
    assert validated["source_suffix_access_authorized"] is False
    assert validated["independent_confirmation_authorized"] is False
    assert validated["claim_authorized"] is False
