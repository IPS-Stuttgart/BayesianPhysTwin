from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_fresh_source_lock import (
    FreshSourceAdmissionConfig,
    build_fresh_cohort_lock,
    build_fresh_source_admission,
    build_object_exclusion_manifest,
    object_exclusion_hash,
    validate_fresh_cohort_lock,
    validate_fresh_source_admission,
    validate_object_exclusion_manifest,
)
from bayesian_phystwin.deform360_official_parity import (
    build_public_parity_contract,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: dict[str, object], digest_key: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_bundle(
    root: Path,
    *,
    object_id: str = "200-test-cloth",
    episode_id: int = 0,
    bimanual: str = "no",
    vertex_count: int = 128,
    frame_len: int = 76,
    active_frame_count: int = 76,
    cameras: tuple[str, ...] = ("cam0", "cam1", "cam2"),
) -> tuple[Path, Path]:
    episode = root / object_id / f"episode_{episode_id:04d}"
    episode.mkdir(parents=True)
    metadata = root / object_id / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "object": object_id,
                "sequences": {
                    str(episode_id): {
                        "bimanual": bimanual,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (episode / "calibrate.pkl").write_bytes(b"not-loaded-calibration")
    (episode / "final_data.pkl").write_bytes(
        b"this is deliberately not a valid pickle and must never be loaded"
    )
    (episode / "start_obj_pcd.ply").write_bytes(
        (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
    )
    split = {
        "frame_len": frame_len,
        "train": [0, int(0.8 * frame_len)],
        "test": [int(0.8 * frame_len), frame_len],
    }
    (episode / "split.json").write_text(json.dumps(split), encoding="utf-8")
    control_meta = {
        "schema": "deform360.processing/control-points/v1",
        "inputs": {
            "robot_sha256": "1" * 64,
            "pcd_sha256": "2" * 64,
            "tactile_sha256": {"sensor": "3" * 64},
        },
        "outputs": {
            "calibrate_sha256": _sha256(episode / "calibrate.pkl"),
            "start_ply_sha256": _sha256(episode / "start_obj_pcd.ply"),
            "split_sha256": _sha256(episode / "split.json"),
            "final_data_sha256": _sha256(episode / "final_data.pkl"),
            "contact_start_frame": 0,
            "contact_end_frame": frame_len - 1,
            "num_active_frames": active_frame_count,
        },
        "parameters": {
            "cameras": list(cameras),
            "contact_threshold": 0.0,
            "contact_patience": 5,
            "taxels_per_gripper": 768,
            "train_fraction": 0.8,
        },
    }
    (episode / "control_points.meta.json").write_text(
        json.dumps(control_meta), encoding="utf-8"
    )
    return episode, metadata


def _admit(
    root: Path,
    *,
    object_id: str = "200-test-cloth",
    episode_id: int = 0,
    category: str = "cloth",
    **bundle_options: object,
) -> dict[str, object]:
    episode, metadata = _write_bundle(
        root,
        object_id=object_id,
        episode_id=episode_id,
        **bundle_options,
    )
    return build_fresh_source_admission(
        episode,
        metadata,
        object_id=object_id,
        episode_id=episode_id,
        category=category,
    )


def test_valid_source_is_admitted_without_deserializing_future_pickle(
    tmp_path: Path,
) -> None:
    artifact = _admit(tmp_path)

    validate_fresh_source_admission(artifact)
    assert artifact["accepted"] is True
    assert artifact["rejection_reasons"] == []
    assert artifact["observed_source_contract"]["frame_zero_point_count"] == 128
    assert artifact["information_boundary"] == {
        "future_object_positions_deserialized": False,
        "future_payload_bytes_hashed": True,
        "future_metrics_read": False,
        "selection_inputs": (
            "raw object/episode identity and enums, stage input/output "
            "provenance hashes/counts, contact-window and split indices, "
            "camera names, and frame-zero PLY vertex count only"
        ),
    }


def test_malformed_bimanual_enum_is_rejected(tmp_path: Path) -> None:
    artifact = _admit(tmp_path, bimanual="yess")

    assert artifact["accepted"] is False
    assert "bimanual must be exactly" in artifact["rejection_reasons"][0]
    validate_fresh_source_admission(artifact)


def test_raw_metadata_must_bind_the_requested_object(tmp_path: Path) -> None:
    episode, metadata = _write_bundle(tmp_path)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["object"] = "201-other-cloth"
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    artifact = build_fresh_source_admission(
        episode,
        metadata,
        object_id="200-test-cloth",
        episode_id=0,
        category="cloth",
    )

    assert artifact["accepted"] is False
    assert (
        "requested object ID is inconsistent with raw metadata"
        in artifact["rejection_reasons"]
    )


def test_split_must_index_actual_final_data_rows(tmp_path: Path) -> None:
    artifact = _admit(tmp_path, active_frame_count=74)

    assert artifact["accepted"] is False
    assert (
        "split indexes the undropped contact window rather than final_data"
        in artifact["rejection_reasons"]
    )


def test_frame_zero_point_count_must_meet_backend_contract(tmp_path: Path) -> None:
    artifact = _admit(tmp_path, vertex_count=54)

    assert artifact["accepted"] is False
    assert (
        "frame-zero point count is outside backend admission"
        in artifact["rejection_reasons"]
    )


def test_camera_panel_and_provenance_are_fail_closed(tmp_path: Path) -> None:
    episode, metadata = _write_bundle(tmp_path, cameras=("cam0", "cam1"))
    (episode / "final_data.pkl").write_bytes(b"changed after provenance")

    artifact = build_fresh_source_admission(
        episode,
        metadata,
        object_id="200-test-cloth",
        episode_id=0,
        category="cloth",
    )

    assert artifact["accepted"] is False
    assert (
        "camera panel is below the preregistered minimum"
        in artifact["rejection_reasons"]
    )
    assert any(
        reason.startswith("future_payload checksum differs")
        for reason in artifact["rejection_reasons"]
    )


def test_frozen_update_must_remain_inside_train_prefix(tmp_path: Path) -> None:
    episode, metadata = _write_bundle(tmp_path)

    artifact = build_fresh_source_admission(
        episode,
        metadata,
        object_id="200-test-cloth",
        episode_id=0,
        category="cloth",
        config=FreshSourceAdmissionConfig(update_frames=(19, 38, 60)),
    )

    assert artifact["accepted"] is False
    assert (
        "a frozen online update falls outside the train prefix"
        in artifact["rejection_reasons"]
    )


def test_resealed_admission_cannot_change_the_frozen_source_contract(
    tmp_path: Path,
) -> None:
    artifact = _admit(tmp_path)
    artifact["config"]["minimum_point_count"] = 54
    artifact["admission_sha256"] = _canonical_sha256(artifact, "admission_sha256")

    with pytest.raises(ValueError, match="changed the frozen config"):
        validate_fresh_source_admission(artifact)


def test_exclusion_manifest_emits_only_namespaced_hashes() -> None:
    artifact = build_object_exclusion_manifest(
        ["002-rope-silk", "083-blanket-cloth"],
        owner="independent-evaluation-owner",
        source_artifact_sha256s=["a" * 64],
    )

    validate_object_exclusion_manifest(artifact)
    encoded = json.dumps(artifact)
    assert "002-rope-silk" not in encoded
    assert "083-blanket-cloth" not in encoded
    assert object_exclusion_hash("002-rope-silk") in artifact["object_hashes"]

    changed = dict(artifact)
    changed["owner"] = "changed"
    with pytest.raises(ValueError, match="checksum changed"):
        validate_object_exclusion_manifest(changed)

    with pytest.raises(ValueError, match="duplicate exclusion"):
        build_object_exclusion_manifest(
            ["002-rope-silk", "002-rope-silk"],
            owner="independent-evaluation-owner",
            source_artifact_sha256s=["a" * 64],
        )


def test_cohort_lock_is_deterministic_and_excludes_reserved_objects(
    tmp_path: Path,
) -> None:
    admissions = [
        _admit(tmp_path / "a", object_id="201-alpha-cloth", category="cloth"),
        _admit(tmp_path / "b", object_id="202-beta-cloth", category="cloth"),
        _admit(tmp_path / "c", object_id="203-gamma-rope", category="rope"),
        _admit(tmp_path / "d", object_id="204-delta-toy", category="toy"),
    ]
    exclusion = build_object_exclusion_manifest(
        ["202-beta-cloth"],
        owner="held-owner",
        source_artifact_sha256s=["b" * 64],
    )
    kwargs = {
        "cohort_size": 3,
        "method_commit": "c" * 40,
        "method_config_sha256": "d" * 64,
        "parity_contract": build_public_parity_contract("per_episode"),
    }

    first = build_fresh_cohort_lock(admissions, [exclusion], **kwargs)
    second = build_fresh_cohort_lock(list(reversed(admissions)), [exclusion], **kwargs)

    validate_fresh_cohort_lock(first)
    assert first == second
    assert [case["object_id"] for case in first["cases"]] == [
        "201-alpha-cloth",
        "203-gamma-rope",
        "204-delta-toy",
    ]
    assert first["exclusion_manifests"] == sorted(first["exclusion_manifests"])
    assert (
        first["evaluation"]["allowed_claim_label"]
        == "fresh_object_candidate_conventions_only"
    )


def test_cohort_lock_rejects_malformed_method_identity(tmp_path: Path) -> None:
    admission = _admit(tmp_path)
    exclusion = build_object_exclusion_manifest(
        ["999-reserved"],
        owner="owner",
        source_artifact_sha256s=["f" * 64],
    )

    with pytest.raises(ValueError, match="method commit is malformed"):
        build_fresh_cohort_lock(
            [admission],
            [exclusion],
            cohort_size=2,
            method_commit="not-a-commit",
            method_config_sha256="d" * 64,
            parity_contract=build_public_parity_contract("per_episode"),
        )


def test_cohort_lock_audits_parity_contract_instead_of_trusting_a_flag(
    tmp_path: Path,
) -> None:
    admission = _admit(tmp_path)
    exclusion = build_object_exclusion_manifest(
        ["999-reserved"],
        owner="owner",
        source_artifact_sha256s=["f" * 64],
    )
    contract = build_public_parity_contract("per_episode")
    contract["fields"]["length_unit"]["value"] = "tampered"

    with pytest.raises(ValueError, match="contract checksum changed"):
        build_fresh_cohort_lock(
            [admission],
            [exclusion],
            cohort_size=2,
            method_commit="c" * 40,
            method_config_sha256="d" * 64,
            parity_contract=contract,
        )


def test_cohort_lock_rejects_conflicting_object_categories(tmp_path: Path) -> None:
    first = _admit(
        tmp_path / "first",
        object_id="205-ambiguous",
        episode_id=0,
        category="cloth",
    )
    second = _admit(
        tmp_path / "second",
        object_id="205-ambiguous",
        episode_id=1,
        category="toy",
    )
    exclusion = build_object_exclusion_manifest(
        ["999-reserved"],
        owner="owner",
        source_artifact_sha256s=["f" * 64],
    )

    with pytest.raises(ValueError, match="conflicting category"):
        build_fresh_cohort_lock(
            [first, second],
            [exclusion],
            cohort_size=2,
            method_commit="c" * 40,
            method_config_sha256="d" * 64,
            parity_contract=build_public_parity_contract("per_episode"),
        )
