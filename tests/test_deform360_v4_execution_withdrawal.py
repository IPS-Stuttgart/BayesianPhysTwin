from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "scripts" / "held" / "seal_deform360_v4_execution_withdrawal.py"


def _operator_module():
    spec = importlib.util.spec_from_file_location(
        "held_v4_withdrawal_operator", OPERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_execution_withdrawal_report_is_exact_and_conservative() -> None:
    module = _operator_module()
    unsigned = module.expected_unsigned_report()
    signed, payload = module._artifact(unsigned)

    assert signed["artifact_sha256"] == (
        "72fb6ba1c6f113157e8351f5b470bcc03d4b20ccec64b3504c7e356fc69cfdc0"
    )
    assert hashlib.sha256(payload).hexdigest() == (
        "9b585f1340a47c64d787a5489faa1c67738d733d4d577648ccb753361c5dd4ca"
    )
    assert unsigned["disposition"] == (
        "WITHDRAWN_AFTER_FRAME_ZERO_BEFORE_PHYSICAL_PREDICTION"
    )
    assert unsigned["cause"]["failure_time_observed_pip_freeze_sha256"] == (
        "NOT_RECORDED_BY_V4"
    )
    assert (
        "not claimed as a reconstruction"
        in unsigned["cause"]["post_failure_diagnostic"]["interpretation"]
    )

    counts = unsigned["execution_counts"]
    assert counts["calibration_lock_count"] == 1
    assert counts["case_attempt_count"] == 2
    assert counts["deployed_snapshot_count"] == 1
    assert counts["deployment_count"] == 1
    assert counts["frame_zero_manifest_count"] == 2
    assert counts["physical_builder_invocation_count"] == 2
    for key in (
        "calibration_decision_count",
        "confirmation_lock_count",
        "formal_online_prediction_count",
        "formal_physical_prediction_count",
        "online_prediction_seal_count",
        "outcome_api_operation_count",
        "outcome_created_count",
        "outcome_permit_count",
        "outcome_read_count",
        "physical_prediction_artifact_count",
        "physical_prior_seal_count",
        "prefix_authorization_count",
        "target_operation_count",
    ):
        assert counts[key] == 0

    boundary = unsigned["information_boundary"]
    assert boundary["episode_payload_read"] is True
    assert boundary["episode_payload_read_scope"] == (
        "frame-zero RGB-D and masks; the frame-zero pipeline read the full "
        "realized robot archive to select the window, then sealed the aligned "
        "76-frame robot-kinematics window"
    )
    for key in (
        "confirmation_payload_read",
        "future_tactile_read",
        "object_future_depth_read",
        "object_future_rgb_read",
        "object_future_tracking_read",
        "prediction_payload_read",
        "tactile_read",
        "target_data_read",
    ):
        assert boundary[key] is False

    paths = {row["path"] for row in unsigned["evidence"]["file_inventory"]}
    assert len(paths) == 19
    assert not any("physical_prior_seal" in path for path in paths)
    assert not any("online_prediction_seal" in path for path in paths)
    assert not any("outcome" in path for path in paths)
    assert unsigned["reuse"] == {
        "v4_frame_zero_artifacts_reused_by_v5": False,
        "v4_physical_or_online_predictions_reused_by_v5": False,
        "v4_formal_physical_or_online_prediction_count_available_for_reuse": 0,
        "v5_requires_fresh_absent_held_root": True,
    }


def test_v4_withdrawal_operator_names_only_the_frozen_evidence_paths() -> None:
    module = _operator_module()
    calibration_files = {
        path for path in module._EXPECTED_FILES if path.startswith("calibration/")
    }
    assert len(calibration_files) == 14
    assert calibration_files | module._EXPECTED_DIRECTORIES == {
        "calibration/.shard-0.claim",
        "calibration/.shard-1.claim",
        "calibration/cases",
        "calibration/cases/002-rope-silk-ep0003",
        "calibration/cases/002-rope-silk-ep0003/frame-zero",
        "calibration/cases/002-rope-silk-ep0003/frame-zero/frame_zero_bundle.manifest.json",
        "calibration/cases/002-rope-silk-ep0003/frame-zero/frame_zero_bundle.npz",
        "calibration/cases/002-rope-silk-ep0003/frame-zero/known_action_76.npz",
        "calibration/cases/083-blanket-cloth-ep0003",
        "calibration/cases/083-blanket-cloth-ep0003/frame-zero",
        "calibration/cases/083-blanket-cloth-ep0003/frame-zero/frame_zero_bundle.manifest.json",
        "calibration/cases/083-blanket-cloth-ep0003/frame-zero/frame_zero_bundle.npz",
        "calibration/cases/083-blanket-cloth-ep0003/frame-zero/known_action_76.npz",
        "calibration/logs",
        "calibration/logs/002-rope-silk-ep0003.frame-zero-validate.log",
        "calibration/logs/002-rope-silk-ep0003.frame-zero.log",
        "calibration/logs/002-rope-silk-ep0003.physical.failed.log",
        "calibration/logs/083-blanket-cloth-ep0003.frame-zero-validate.log",
        "calibration/logs/083-blanket-cloth-ep0003.frame-zero.log",
        "calibration/logs/083-blanket-cloth-ep0003.physical.failed.log",
        "calibration/logs/shard-0.lock-verification.log",
        "calibration/logs/shard-1.lock-verification.log",
    }
