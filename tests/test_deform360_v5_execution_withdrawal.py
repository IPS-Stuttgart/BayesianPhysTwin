from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import stat

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "scripts" / "held" / "seal_deform360_v5_execution_withdrawal.py"


def _operator_module():
    spec = importlib.util.spec_from_file_location(
        "held_v5_withdrawal_operator", OPERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_execution_withdrawal_report_is_exact_and_conservative() -> None:
    module = _operator_module()
    unsigned = module.expected_unsigned_report()
    signed, payload = module._artifact(unsigned)

    assert signed["artifact_sha256"] == (
        "38a2f0b3c08ccac82b3d301fb77b993b7c994ba5f42c260c2a669a0d80a57def"
    )
    assert hashlib.sha256(payload).hexdigest() == (
        "915a556fb8a412b314766c7169878db9880200f63d07dd3ec82476be5d2ea14b"
    )
    assert unsigned["disposition"] == (
        "WITHDRAWN_DURING_FIRST_TARGET_OPERATION_BEFORE_ANY_COMPLETED_OUTCOME"
    )
    assert unsigned["cause"] == {
        "classification": "PROPAGATED_FRAME_ZERO_MASK_SEAL_MISMATCH",
        "exception_message": (
            "propagated frame-zero mask differs from seal: brics-odroid-001_cam0"
        ),
        "failed_camera": "brics-odroid-001_cam0",
        "failed_case": "002-rope-silk-ep0003",
        "failure_phase": "first calibration target reconstruction operation",
        "terminal_log": {
            "path": "calibration-outcomes.console.log",
            "sha256": (
                "7f48d2ee1291d37f051a9422e2217fafe7f08d7438285b8b4f4ee14e5d93ab71"
            ),
            "size": 3270,
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
    assert counts["target_operation_failed_count"] == 1
    assert counts["partial_target_staging_file_count"] == 28
    assert counts["partial_target_staging_directory_count"] == 12
    assert counts["staged_camera_video_count"] == 8
    for key in (
        "calibration_decision_count",
        "calibration_score_evidence_count",
        "confirmation_case_execution_count",
        "confirmation_lock_count",
        "confirmation_prediction_seal_count",
        "outcome_created_count",
        "outcome_read_count",
        "target_operation_completed_count",
        "target_reconstruction_artifact_count",
    ):
        assert counts[key] == 0

    boundary = unsigned["information_boundary"]
    assert boundary["object_future_rgb_read"] == (
        "POSSIBLE_WITHIN_FIRST_CALIBRATION_CASE_ONLY"
    )
    assert boundary["object_future_rgb_read_case_upper_bound"] == 1
    assert boundary["partial_target_source_staging_created"] is True
    assert boundary[
        "first_case_online_prediction_arrays_decoded_before_target_callback"
    ]
    assert boundary["later_case_online_prediction_arrays_decoded"] is False
    assert boundary["confirmation_payload_read"] is False
    assert boundary["calibration_gate_or_metric_created_or_read"] is False
    assert (
        boundary[
            "forensic_audit_disclosed_arrays_images_masks_metrics_or_protected_values"
        ]
        is False
    )
    for key in (
        "future_tactile_read",
        "object_future_depth_read",
        "object_future_tracking_read",
        "official_target_reconstruction_created",
        "tactile_read",
    ):
        assert boundary[key] is False

    assert unsigned["reuse"] == {
        "v5_evidence_may_be_used_by_v6_only_as_immutable_lineage": True,
        "v5_execution_artifacts_reused_by_v6": False,
        "v5_partial_target_staging_reused_by_v6": False,
        "v5_physical_or_online_predictions_reused_by_v6": False,
        "v5_score_or_gate_available_for_reuse": False,
        "v6_requires_fresh_absent_held_root": True,
        "v6_requires_fresh_predictions_and_outcome_phase": True,
    }


def test_v5_withdrawal_operator_names_the_exact_metadata_only_inventory() -> None:
    module = _operator_module()
    directories, files = module._expected_paths()

    assert len(directories) == 124
    assert len(files) == 574
    assert len(module._EXPECTED_PARTIAL_OUTCOME_FILES) == 28
    assert (
        sum(size for size, _ in module._EXPECTED_PARTIAL_OUTCOME_FILES.values())
        == 176_058_588
    )
    assert sum(path.endswith("/undistorted.mp4") for path in files) == 8
    assert not any("mask_refined" in path for path in files)
    assert not any("calibration-score-evidence" in path for path in files)
    assert not any("calibration-gate-decision" in path for path in files)
    assert not any("object_points" in path for path in files)

    source = OPERATOR.read_text(encoding="utf-8")
    for forbidden in (
        "json.loads(",
        "np.load(",
        "numpy",
        "cv2",
        "VideoCapture",
        "read_text(",
        "read_bytes(",
    ):
        assert forbidden not in source


def test_v5_withdrawal_inventory_refuses_extra_evidence_and_seals_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _operator_module()
    held = tmp_path / "held-v5"
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
    monkeypatch.setattr(module, "_REPORT", held / "v5-outcome-withdrawal-report.json")
    monkeypatch.setattr(module, "_EXPECTED_ROOT_FILES", expected_root)
    monkeypatch.setattr(module, "_EXPECTED_PARTIAL_OUTCOME_FILES", {})
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
        RuntimeError, match="unexpected held-v5 evidence file inventory"
    ):
        module._inventory()
    unexpected.unlink()

    try:
        module.main()
        report = held / "v5-outcome-withdrawal-report.json"
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
