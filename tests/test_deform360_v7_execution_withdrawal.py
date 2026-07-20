from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "scripts" / "held" / "seal_deform360_v7_execution_withdrawal.py"


def _operator_module():
    spec = importlib.util.spec_from_file_location(
        "held_v7_withdrawal_operator", OPERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v7_execution_withdrawal_report_is_exact_and_conservative() -> None:
    module = _operator_module()
    unsigned = module.expected_unsigned_report()
    signed, payload = module._artifact(unsigned)

    assert len(payload) == 10_295
    assert signed["artifact_sha256"] == (
        "8752e25922a4604222c2ce2cdcf9f14c84f661954119b39c01c5f2c12bd4231f"
    )
    assert hashlib.sha256(payload).hexdigest() == (
        "7bcab7169fc2addad8e56b7bb5ca9086b5249e9a744e18b9d51a7f395098c1a3"
    )
    assert unsigned["protocol_id"] == "deform360-held-online-belief-v7"
    assert unsigned["replacement_protocol_id"] == ("deform360-held-online-belief-v8")
    assert unsigned["result_status"] == "NO_CALIBRATION_RESULT"
    assert unsigned["disposition"] == (
        "WITHDRAWN_AFTER_FIRST_COMPLETED_TARGET_BEFORE_ANY_COMPLETED_CASE_SCORE"
    )
    assert unsigned["cause"] == {
        "cardinality_relation_disclosed_by_terminal_failure": (
            "eligible visible-and-valid official frame-zero identity count "
            "is less than sealed frame-zero point count"
        ),
        "classification": ("INSUFFICIENT_VISIBLE_VALID_OFFICIAL_FRAME_ZERO_IDENTITIES"),
        "exception_message": (
            "too few visible and valid official frame-zero identities"
        ),
        "exception_type": "ValueError",
        "failed_case": "002-rope-silk-ep0003",
        "failure_phase": (
            "first calibration case identity-transport eligibility-cardinality "
            "precondition after completed CREATE target operation and before "
            "sparse assignment or metric computation"
        ),
        "terminal_exit_code": 2,
        "terminal_log": {
            "path": "calibration-outcomes.console.log",
            "sha256": (
                "debdfd4267cbf814e8d87cbcd55c857fed375599c36810fe1908a039802f136d"
            ),
            "size": 130_067_916,
        },
    }

    counts = unsigned["execution_counts"]
    for key in (
        "formal_online_prediction_count",
        "formal_physical_prediction_count",
        "frame_zero_bundle_count",
        "frame_zero_manifest_count",
        "online_prediction_seal_count",
        "physical_prior_seal_count",
        "prefix_authorization_count",
        "target_operation_planned_count",
    ):
        assert counts[key] == 15
    assert counts["outcome_permit_count"] == 1
    assert counts["target_operation_started_count"] == 1
    assert counts["target_operation_completed_count"] == 1
    assert counts["target_operation_failed_count"] == 0
    assert counts["target_reconstruction_artifact_count"] == 1
    assert counts["target_reconstruction_completed_count"] == 1
    assert counts["completed_target_staging_directory_count"] == 21
    assert counts["completed_target_staging_file_count"] == 235
    assert counts["pcd_clean_frame_archive_count"] == 76
    assert counts["splatfacto_ply_count"] == 81
    assert counts["identity_transport_attempted_count"] == 1
    assert counts["identity_transport_completed_count"] == 0
    for key in (
        "calibration_case_score_completed_count",
        "calibration_decision_count",
        "calibration_score_evidence_count",
        "confirmation_case_execution_count",
        "confirmation_lock_count",
        "confirmation_prediction_seal_count",
        "metric_computation_started_count",
        "outcome_read_count",
        "sparse_identity_assignment_started_count",
    ):
        assert counts[key] == 0

    boundary = unsigned["information_boundary"]
    assert boundary["first_case_official_target_arrays_constructed"] is True
    assert boundary["first_case_identity_eligibility_relation_evaluated"] is True
    assert boundary["official_target_reconstruction_created"] is True
    assert boundary["object_future_mask_downstream_read"] is True
    assert boundary["rendered_future_depth_archive_count"] == 8
    assert boundary["rendered_future_depth_downstream_read"] is True
    assert boundary["derived_future_tracking_downstream_read"] is True
    assert boundary["derived_future_tracking_velocity_archive_count"] == 8
    assert boundary["derived_future_tracking_visibility_archive_count"] == 8
    assert boundary["source_dataset_future_depth_read"] is False
    assert boundary["source_dataset_future_tracking_read"] is False
    assert boundary["sparse_identity_assignment_created"] is False
    assert boundary["exact_identity_cardinalities_in_withdrawal_report"] is False
    assert boundary["calibration_gate_or_metric_created_or_read"] is False
    assert boundary["confirmation_payload_read"] is False
    assert boundary["later_case_online_prediction_arrays_decoded"] is False
    assert (
        boundary[
            "forensic_audit_disclosed_arrays_images_masks_metrics_or_protected_values"
        ]
        is False
    )

    evidence = unsigned["evidence"]
    assert evidence["complete_noncode_inventory"] == {
        "directory_count": 134,
        "file_count": 784,
        "inventory_entry_count": 918,
        "inventory_sha256": (
            "6e7c639455963fcf807685525c028c24955ce7ab8884d8daa02b2f91b3696e7f"
        ),
        "total_file_bytes": 1_010_473_211,
    }
    assert evidence["structured_terminal_event_counts"] == {
        "calibration_cohort_barrier_validated": 1,
        "calibration_gate_decision_written": 0,
        "calibration_score_evidence_written": 0,
        "calibration_target_operation_complete": 1,
        "calibration_target_operation_planned": 15,
        "calibration_target_operation_start": 1,
        "fail_closed": 1,
        "gsplat_runtime_smoke_validated": 1,
    }
    assert unsigned["reuse"] == {
        "v7_completed_target_or_staging_reused_by_v8": False,
        "v7_evidence_may_be_used_by_v8_only_as_immutable_lineage": True,
        "v7_execution_artifacts_reused_by_v8": False,
        "v7_physical_or_online_predictions_reused_by_v8": False,
        "v7_score_or_gate_available_for_reuse": False,
        "v8_requires_fresh_absent_held_root": True,
        "v8_requires_fresh_predictions_and_outcome_phase": True,
    }


def test_v7_withdrawal_operator_binds_only_the_metadata_inventory() -> None:
    module = _operator_module()
    directories, files = module._expected_paths()
    outcome_directories, outcome_files = module._expected_outcome_paths()

    assert len(directories) == 134
    assert len(files) == 784
    assert len(outcome_directories) == 22
    assert len(outcome_files) == 237
    assert len(module._EXPECTED_KEY_OUTCOME_FILES) == 4
    assert sum(path.endswith("/undistorted.mp4") for path in outcome_files) == 8
    assert sum(path.endswith("/mask_refined.h5") for path in outcome_files) == 8
    assert sum(path.endswith("/rendered_depth.h5") for path in outcome_files) == 8
    assert sum("/tracking/vel.h5" in path for path in outcome_files) == 8
    assert sum("/tracking/visibility.h5" in path for path in outcome_files) == 8
    assert sum("/pcd_clean/" in path and path.endswith(".npz") for path in files) == 76
    assert sum("/splatfacto/splat_" in path for path in files) == 81
    assert not any("calibration-score-evidence" in path for path in files)
    assert not any("calibration-gate-decision" in path for path in files)
    assert not any("confirmation" in path for path in directories | files)

    source = OPERATOR.read_text(encoding="utf-8")
    for forbidden in (
        "json.loads(",
        "np.load(",
        "numpy",
        "h5py",
        "cv2",
        "PIL",
        "imageio",
        "VideoCapture",
        "read_text(",
        "read_bytes(",
    ):
        assert forbidden not in source


def test_v7_withdrawal_inventory_refuses_extra_evidence_and_seals_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _operator_module()
    temporary = tempfile.TemporaryDirectory(
        prefix="bpt-v7-withdrawal-test-", dir="/tmp"
    )
    held = Path(temporary.name) / "held-v7"
    calibration = held / "calibration"
    code = held / module._CODE_NAME
    calibration.mkdir(parents=True)
    code.mkdir()
    code.chmod(0o555)
    root_contents = {
        "calibration-lock.json": b"lock\n",
        "calibration-outcomes.console.log": b"failure\n",
        "calibration-shard-0.console.log": b"shard zero\n",
        "calibration-shard-1.console.log": b"shard one\n",
        "gsplat-runtime-smoke-evidence.json": b"smoke\n",
    }
    for name, payload in root_contents.items():
        (held / name).write_bytes(payload)

    expected_root = {
        name: (len(payload), hashlib.sha256(payload).hexdigest())
        for name, payload in root_contents.items()
    }
    rows = [{"path": "calibration", "type": "directory"}]
    rows.extend(
        {
            "path": name,
            "sha256": sha256,
            "size": size,
            "type": "file",
        }
        for name, (size, sha256) in sorted(expected_root.items())
    )
    rows.sort(key=lambda row: str(row["path"]))

    monkeypatch.setattr(module, "_HELD_ROOT", held)
    monkeypatch.setattr(module, "_REPORT", held / "v7-outcome-withdrawal-report.json")
    monkeypatch.setattr(module, "_EXPECTED_ROOT_FILES", expected_root)
    monkeypatch.setattr(module, "_EXPECTED_KEY_OUTCOME_FILES", {})
    monkeypatch.setattr(module, "_EXPECTED_EVIDENCE", module._summary(rows))
    monkeypatch.setattr(module, "_EXPECTED_CATEGORIES", {})
    monkeypatch.setattr(
        module,
        "_expected_paths",
        lambda: ({"calibration"}, set(root_contents)),
    )
    monkeypatch.setattr(module.socket, "gethostname", lambda: "workstation2")

    assert module._inventory() == rows
    unexpected = calibration / "unexpected.bin"
    unexpected.write_bytes(b"not allowed")
    with pytest.raises(
        RuntimeError, match="unexpected held-v7 evidence file inventory"
    ):
        module._inventory()
    unexpected.unlink()

    try:
        module.main()
        report = held / "v7-outcome-withdrawal-report.json"
        assert stat.S_IMODE(report.stat().st_mode) == 0o400
        assert stat.S_IMODE(held.stat().st_mode) == 0o500
        assert stat.S_IMODE(calibration.stat().st_mode) == 0o500
        assert stat.S_IMODE(code.stat().st_mode) == 0o555
        for name in root_contents:
            assert stat.S_IMODE((held / name).stat().st_mode) == 0o400
        first_digest = hashlib.sha256(report.read_bytes()).hexdigest()
        module.main()
        assert hashlib.sha256(report.read_bytes()).hexdigest() == first_digest
    finally:
        os.chmod(held, 0o700)
        os.chmod(calibration, 0o700)
        os.chmod(code, 0o700)
        for path in held.iterdir():
            if path.is_file():
                os.chmod(path, 0o600)
        temporary.cleanup()
