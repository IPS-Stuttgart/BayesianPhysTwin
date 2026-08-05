from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_visual_provider_recovery_lock import (
    DEFORM360_FUTURE_FRAMES,
    DEFORM360_OBSERVED_CONTACT_FRAMES,
    DEFORM360_OBSERVED_HISTORY_FRAMES,
    DEFORM360_PROVIDER_OVERLAP,
    DEFORM360_PROVIDER_WINDOW_COUNT,
    DEFORM360_PROVIDER_WINDOW_SIZE,
    Deform360VisualProviderRecoveryLockV1,
    derive_deform360_causal_window,
    first_deform360_contact_frame,
    load_deform360_visual_provider_recovery_lock,
    save_deform360_visual_provider_recovery_lock,
    select_deform360_camera_panel,
)
from bayesian_phystwin.prob4d_provider_attestation import (
    compute_prob4d_provider_manifest_id,
)


def _lock(**updates: object) -> Deform360VisualProviderRecoveryLockV1:
    values: dict[str, object] = {
        "provider_revision": "1" * 40,
        "provider_manifest_id": "2" * 64,
        "provider_manifest_sha256": "3" * 64,
        "motioncrafter_revision": "4" * 40,
        "motioncrafter_model_set_id": "5" * 64,
        "motioncrafter_model_set_manifest_sha256": "6" * 64,
        "initial_metric_frame_prior_policy_id": "7" * 64,
        "metadata": {"information_order": "post-payload-pre-score"},
    }
    values.update(updates)
    return Deform360VisualProviderRecoveryLockV1(**values)  # type: ignore[arg-type]


def _tactile(frame_count: int, contact_frame: int | None) -> dict[str, np.ndarray]:
    streams = {
        "brics-odroid_tactilel_left": np.zeros((frame_count, 16, 32), dtype=np.float32),
        "brics-odroid_tactilel_right": np.zeros(
            (frame_count, 16, 32), dtype=np.float32
        ),
        "brics-odroid_tactiler_left": np.zeros((frame_count, 16, 32), dtype=np.float32),
        "brics-odroid_tactiler_right": np.zeros(
            (frame_count, 16, 32), dtype=np.float32
        ),
    }
    if contact_frame is not None:
        streams["brics-odroid_tactilel_left"][contact_frame, 0, 0] = 0.4
        streams["brics-odroid_tactilel_right"][contact_frame, 0, 1] = 0.4
    return streams


def test_recovery_lock_is_content_addressed_and_roundtrips(tmp_path) -> None:
    lock = _lock()
    path = tmp_path / "recovery-lock.json"
    save_deform360_visual_provider_recovery_lock(path, lock)
    loaded = load_deform360_visual_provider_recovery_lock(path)

    assert loaded == lock
    assert loaded.artifact_id == lock.artifact_id
    assert loaded.to_record()["provider_api_version"] == 2
    assert loaded.to_record()["window_size"] == DEFORM360_PROVIDER_WINDOW_SIZE
    assert loaded.to_record()["overlap"] == DEFORM360_PROVIDER_OVERLAP
    assert loaded.to_record()["window_count"] == DEFORM360_PROVIDER_WINDOW_COUNT
    assert loaded.to_record()["fusion_rule"] == "decoded-uniform"
    assert loaded.to_record()["seed_policy"] == "derived-per-call"
    assert loaded.to_record()["selected_calibration_payloads_opened"] is True
    assert loaded.to_record()["calibration_scores_opened"] is False
    assert loaded.to_record()["confirmation_payloads_opened"] is False


def test_recovery_lock_rejects_retroactive_or_post_score_claims() -> None:
    cases = [
        ({"selected_calibration_payloads_opened": False}, "opened calibration"),
        (
            {"calibration_values_used_for_provider_selection": True},
            "must not use calibration values",
        ),
        ({"calibration_scores_opened": True}, "precede calibration"),
        ({"calibration_policy_fit": True}, "precede calibration"),
        ({"confirmation_payloads_opened": True}, "precede confirmation"),
        ({"target_outcomes_used": True}, "precede confirmation"),
    ]
    for updates, message in cases:
        with pytest.raises(ValueError, match=message):
            _lock(**updates)


def test_recovery_lock_rejects_configuration_and_lineage_drift() -> None:
    cases = [
        {"provider_revision": int("1" * 40)},
        {"provider_manifest_id": int("2" * 64)},
        {"motioncrafter_revision": int("4" * 40)},
        {"initial_metric_frame_prior_policy_id": int("7" * 64)},
        {"motioncrafter_model_type": "diff"},
        {"storage_dtype": "float64"},
        {"additional_metric_anchor_policy": "independent_sparse"},
        {"max_gauge_rank": 0},
        {"minimum_retained_gauge_trace": 0.0},
        {"minimum_retained_gauge_trace": float("nan")},
        {"stage1_provenance_id": "a" * 64},
        {"selection_artifact_sha256": "b" * 64},
        {"provider_repository": "another/Prob4D"},
    ]
    for updates in cases:
        with pytest.raises(ValueError):
            _lock(**updates)


def test_recovery_lock_loader_rejects_semantic_drift_and_tampering(tmp_path) -> None:
    lock = _lock()
    for field_name, invalid in (
        ("provider_api_version", 3),
        ("window_size", 24),
        ("full_joint_gauge_covariance", False),
        ("calibration_scores_opened", 0),
    ):
        record = lock.to_record()
        record[field_name] = invalid
        with pytest.raises(ValueError):
            Deform360VisualProviderRecoveryLockV1.from_mapping(record)

    record = lock.to_record()
    record["height"] = 321
    with pytest.raises(ValueError, match="artifact_id"):
        Deform360VisualProviderRecoveryLockV1.from_mapping(record)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_deform360_visual_provider_recovery_lock(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_deform360_visual_provider_recovery_lock(nonfinite)


def test_first_contact_matches_official_grouped_taxel_rule() -> None:
    streams = _tactile(100, None)
    streams["brics-odroid_tactilel_left"][50, 0, 0] = 0.4
    streams["brics-odroid_tactilel_right"][51, 0, 0] = 0.4
    streams["brics-odroid_tactilel_left"][51, 0, 1] = 0.4

    assert (
        first_deform360_contact_frame(
            streams,
            total_episode_frames=100,
        )
        == 51
    )


def test_contact_search_does_not_read_values_after_first_contact() -> None:
    streams = _tactile(100, 50)
    for values in streams.values():
        values[51:] = np.nan

    assert (
        first_deform360_contact_frame(
            streams,
            total_episode_frames=100,
        )
        == 50
    )


def test_causal_window_has_two_overlap_windows_and_untouched_future() -> None:
    streams = _tactile(150, 60)
    window = derive_deform360_causal_window(
        streams,
        total_episode_frames=150,
    )

    assert window.contact_start_frame == 60
    assert window.causal_cutoff_frame == 60 + DEFORM360_OBSERVED_CONTACT_FRAMES
    assert window.observed_frame_count == DEFORM360_OBSERVED_HISTORY_FRAMES
    assert window.future_frame_count == DEFORM360_FUTURE_FRAMES
    assert window.processing_frame_count == (
        DEFORM360_OBSERVED_HISTORY_FRAMES + DEFORM360_FUTURE_FRAMES
    )
    first = range(window.source_start_frame, window.source_start_frame + 25)
    second = range(window.source_start_frame + 17, window.causal_cutoff_frame)
    assert len(set(first) & set(second)) == DEFORM360_PROVIDER_OVERLAP


def test_causal_window_fails_without_contact_or_required_context() -> None:
    with pytest.raises(ValueError, match="no tactile contact"):
        derive_deform360_causal_window(
            _tactile(100, None),
            total_episode_frames=100,
        )
    with pytest.raises(ValueError, match="pre-contact history"):
        derive_deform360_causal_window(
            _tactile(100, 10),
            total_episode_frames=100,
        )
    with pytest.raises(ValueError, match="untouched future"):
        derive_deform360_causal_window(
            _tactile(70, 50),
            total_episode_frames=70,
        )


def test_camera_panel_uses_only_pose_geometry_with_lexicographic_ties() -> None:
    poses: dict[str, np.ndarray] = {}
    for name, center in {
        "camera-a": (1.0, 0.0, 0.0),
        "camera-b": (0.0, 1.0, 0.0),
        "camera-c": (-1.0, 0.0, 0.0),
        "camera-d": (0.0, -1.0, 0.0),
    }.items():
        pose = np.eye(4)
        pose[:3, 3] = center
        poses[name] = pose

    assert select_deform360_camera_panel(poses) == (
        "camera-a",
        "camera-b",
        "camera-c",
    )


def test_camera_panel_rejects_invalid_or_insufficient_calibration() -> None:
    with pytest.raises(ValueError, match="insufficient calibrated"):
        select_deform360_camera_panel({"a": np.eye(4), "b": np.eye(4)})

    invalid = {name: np.eye(4) for name in ("a", "b", "c")}
    invalid["c"][3, 3] = 0.0
    with pytest.raises(ValueError, match="homogeneous"):
        select_deform360_camera_panel(invalid)


def test_recovery_lock_record_is_plain_finite_json() -> None:
    lock = _lock()
    serialized = json.dumps(lock.to_record(), sort_keys=True, allow_nan=False)
    loaded = json.loads(serialized)
    assert loaded["artifact_id"] == lock.artifact_id


@pytest.mark.parametrize(
    "lock_filename",
    [
        "deform360_official_hub_visuotactile_v1_visual_provider_recovery_v1.json",
        "deform360_official_hub_visuotactile_v2_visual_provider_recovery_v1.json",
    ],
)
def test_committed_recovery_lock_binds_exact_provider_assets(
    lock_filename: str,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    lock = load_deform360_visual_provider_recovery_lock(
        repository / "protocols/locks/" / lock_filename
    )
    provider_path = repository / lock.metadata["prob4d_provider_manifest_path"]
    model_path = repository / lock.metadata["motioncrafter_model_set_manifest_path"]
    metric_policy_path = (
        repository / lock.metadata["initial_metric_frame_prior_policy_path"]
    )

    provider_bytes = provider_path.read_bytes()
    provider = json.loads(provider_bytes)
    assert hashlib.sha256(provider_bytes).hexdigest() == lock.provider_manifest_sha256
    assert compute_prob4d_provider_manifest_id(provider) == lock.provider_manifest_id
    assert provider["provider_revision"] == lock.provider_revision

    model_bytes = model_path.read_bytes()
    model = json.loads(model_bytes)
    assert hashlib.sha256(model_bytes).hexdigest() == (
        lock.motioncrafter_model_set_manifest_sha256
    )
    canonical_model = json.dumps(
        model,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert hashlib.sha256(canonical_model).hexdigest() == (
        lock.motioncrafter_model_set_id
    )

    policy = json.loads(metric_policy_path.read_text(encoding="utf-8"))
    declared_policy_id = policy.pop("artifact_id")
    canonical_policy = json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert hashlib.sha256(canonical_policy).hexdigest() == declared_policy_id
    assert declared_policy_id == lock.initial_metric_frame_prior_policy_id


def test_recovery_amendment_records_corrected_information_order() -> None:
    repository = Path(__file__).resolve().parents[1]
    path = repository / (
        "protocols/amendments/"
        "deform360_official_hub_visuotactile_v1_provider_recovery.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    declared_id = amendment.pop("artifact_id")
    canonical = json.dumps(
        amendment,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()

    assert hashlib.sha256(canonical).hexdigest() == declared_id
    assert amendment["status"] == (
        "locked-post-calibration-payload-pre-calibration-score"
    )
    boundary = amendment["corrected_information_order"]
    assert boundary["calibration_camera_tactile_and_robot_payloads_acquired"] is True
    assert boundary["calibration_scores_opened"] is False
    assert boundary["confirmation_payloads_opened"] is False
    assert boundary["target_outcomes_used"] is False


def test_visual_execution_lock_binds_target_free_camera_panel() -> None:
    repository = Path(__file__).resolve().parents[1]
    execution_path = repository / (
        "protocols/locks/"
        "deform360_official_hub_visuotactile_v1_visual_execution_lock_v1.json"
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution_id = execution.pop("artifact_id")
    assert (
        hashlib.sha256(
            json.dumps(
                execution,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        == execution_id
    )

    panel_path = repository / execution["camera_panel_policy"]["path"]
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel_id = panel.pop("artifact_id")
    assert (
        hashlib.sha256(
            json.dumps(
                panel,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        == panel_id
    )
    assert panel_id == execution["camera_panel_policy"]["artifact_id"]
    assert panel["panel_size"] == 3
    assert panel["candidate_contract"]["image_values_used"] is False
    assert panel["candidate_contract"]["outcomes_used"] is False

    provider_path = repository / execution["visual_provider_recovery_lock"]["path"]
    provider = load_deform360_visual_provider_recovery_lock(provider_path)
    assert (
        provider.artifact_id
        == execution["visual_provider_recovery_lock"]["artifact_id"]
    )
