#!/usr/bin/env python3
"""Seal the terminal held-v7 outcome failure and write its withdrawal report.

This one-purpose forensic operator inventories the complete non-code held-v7
execution tree using only filenames, file metadata, and stable SHA-256 byte
streams.  It does not deserialize a target, prediction, mask, point cloud,
image, video, metric, or other protected payload.  The fixed report represents
only the already-observed terminal error and the structural execution boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


_HELD_ROOT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v7")
_REPORT = _HELD_ROOT / "v7-outcome-withdrawal-report.json"
_CODE_NAME = "code-a5c4bcb463e3eaba066280b811d25382789c1104"
_PROTOCOL_ID = "deform360-held-online-belief-v7"
_REPLACEMENT_PROTOCOL_ID = "deform360-held-online-belief-v8"

_CALIBRATION_CASES = (
    "002-rope-silk-ep0003",
    "002-rope-silk-ep0004",
    "002-rope-silk-ep0008",
    "083-blanket-cloth-ep0000",
    "083-blanket-cloth-ep0003",
    "083-blanket-cloth-ep0006",
    "085-scarf-cloth-ep0000",
    "085-scarf-cloth-ep0005",
    "085-scarf-cloth-ep0007",
    "092-squirrel-ep0002",
    "092-squirrel-ep0003",
    "092-squirrel-ep0006",
    "170-spider-ep0002",
    "170-spider-ep0004",
    "170-spider-ep0007",
)
_FAILED_CASE = _CALIBRATION_CASES[0]
_STAGED_CAMERAS = (
    "brics-odroid-001_cam0",
    "brics-odroid-006_cam0",
    "brics-odroid-007_cam0",
    "brics-odroid-014_cam1",
    "brics-odroid-021_cam1",
    "brics-odroid-025_cam1",
    "brics-odroid-027_cam0",
    "brics-odroid-028_cam0",
)

_CASE_FILE_SUFFIXES = (
    "frame-zero/frame_zero_bundle.manifest.json",
    "frame-zero/frame_zero_bundle.npz",
    "frame-zero/known_action_76.npz",
    "online/measurement.npz",
    "online/measurement_cycle_uncertainty.npz",
    "online/measurement_cycle_uncertainty_manifest.json",
    "online/measurement_manifest.json",
    "online/measurement_uncertainty.npz",
    "online/measurement_uncertainty_manifest.json",
    "online/online_prediction.npz",
    "online/online_prediction_seal.json",
    "physical/episode_graph.npz",
    "physical/logs/automatic_twin.log",
    "physical/logs/warp_driven.log",
    "physical/logs/warp_zero_action.log",
    "physical/physical_prediction_manifest.json",
    "physical/physical_prior_seal.json",
    "physical/prediction.npz",
    "physical/prediction_only_input.json",
    "physical/prediction_only_input.pkl",
    "physical/simulator_final_data.pkl",
    "physical/state_artifact.npz",
    "physical/twin_summary.json",
    "physical/warp_driven/official_phystwin_smoke.json",
    "physical/warp_driven/official_phystwin_trajectory.npz",
    "physical/warp_zero_action/official_phystwin_smoke.json",
    "physical/warp_zero_action/official_phystwin_trajectory.npz",
    "prefix-authorization.json",
)
_CASE_DIRECTORY_SUFFIXES = (
    "",
    "frame-zero",
    "online",
    "physical",
    "physical/logs",
    "physical/warp_driven",
    "physical/warp_zero_action",
)
_CASE_LOG_SUFFIXES = (
    "frame-zero-validate.log",
    "frame-zero.log",
    "online-validate.log",
    "online.log",
    "physical-validate.log",
    "physical.log",
    "prefix-authorization.log",
    "prefix-validate.log",
)

_EXPECTED_ROOT_FILES: Mapping[str, tuple[int, str]] = {
    "calibration-lock.json": (
        17_610,
        "b464d7cfda3b4ad94f57ffd46267b3b50d8dc65e2ff8dfec2befc7953718aca7",
    ),
    "calibration-outcomes.console.log": (
        130_067_916,
        "debdfd4267cbf814e8d87cbcd55c857fed375599c36810fe1908a039802f136d",
    ),
    "calibration-shard-0.console.log": (
        2_723,
        "bddd219eaee0a8aeb2a800e6c41c5a4b9e0b4c1e63500b09baf42e94dbf43264",
    ),
    "calibration-shard-1.console.log": (
        2_383,
        "fc1f903947dd3b29bb6b34c7f934da0be9d4eb602ae0282b6a427e439afb3279",
    ),
    "gsplat-runtime-smoke-evidence.json": (
        3_606,
        "c5f0218962e1c18748f52d423c11804864e2695a719f00ff63452cebdbde029c",
    ),
}

_OUTCOME_CASE_ROOT = f"calibration/outcomes/{_FAILED_CASE}"
_OUTCOME_EPISODE_ROOT = f"{_OUTCOME_CASE_ROOT}/staged-aligned/episode_0000"
_EXPECTED_KEY_OUTCOME_FILES: Mapping[str, tuple[int, str]] = {
    f"{_OUTCOME_CASE_ROOT}/held_outcome.json": (
        228_509,
        "8631a16c9d6308207df2c271ae3e43c6439289b08acac2cafb10f5fb45e33763",
    ),
    f"{_OUTCOME_CASE_ROOT}/official_target.npz": (
        536_992,
        "850a894f1e1eb447fddbb877ac2fbf38225e97514a1218cc7ea1182212f471a8",
    ),
    f"{_OUTCOME_EPISODE_ROOT}/pcd_clean/pcd_clean.meta.json": (
        9_404,
        "887d6b7e7a47c5b9d0919086e4537c71097a18daf120783e3d4b51a00885f3e1",
    ),
    f"{_OUTCOME_EPISODE_ROOT}/splatfacto/splatfacto.meta.json": (
        8_672,
        "17b8a31fa72ef7689e4573bdbdfcedcaf183347e6d09c4b85f8452e3a7b711c0",
    ),
}

_EXPECTED_EVIDENCE = {
    "directory_count": 134,
    "file_count": 784,
    "inventory_entry_count": 918,
    "inventory_sha256": (
        "6e7c639455963fcf807685525c028c24955ce7ab8884d8daa02b2f91b3696e7f"
    ),
    "total_file_bytes": 1_010_473_211,
}
_EXPECTED_CATEGORIES = {
    "case_artifacts": {
        "directory_count": 105,
        "file_count": 420,
        "inventory_entry_count": 525,
        "inventory_sha256": (
            "094a2ec2e01a4e1175a6b82cb201896b7af51db4275b628fa3f9efb46705b199"
        ),
        "path_prefix": "calibration/cases/",
        "total_file_bytes": 644_080_117,
    },
    "completed_first_outcome": {
        "directory_count": 22,
        "file_count": 237,
        "inventory_entry_count": 259,
        "inventory_sha256": (
            "2cc1ded3a77d4230b2ae12eda0b0237f931b36fa6c8769615e88daca1ca85869"
        ),
        "path_prefix": "calibration/outcomes/",
        "total_file_bytes": 229_435_666,
    },
    "execution_logs": {
        "directory_count": 0,
        "file_count": 122,
        "inventory_entry_count": 122,
        "inventory_sha256": (
            "62be9a4dd12348251abb97422e1aca4b6781c1d0b7530ad4e9661c70e241783f"
        ),
        "path_prefix": "calibration/logs/",
        "total_file_bytes": 6_863_190,
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_inventory(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        json.dumps(
            rows,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _artifact(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    unsigned = dict(value)
    digest = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    signed = dict(unsigned)
    signed["artifact_sha256"] = digest
    return signed, _canonical_json(signed)


def _sha256_regular_file(path: Path) -> tuple[int, str]:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
    _require(not path.is_symlink(), f"symlink refused: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            (before.st_dev, before.st_ino, before.st_size)
            == (opened.st_dev, opened.st_ino, opened.st_size),
            f"file changed before open: {path}",
        )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {path}",
    )
    return before.st_size, digest.hexdigest()


def _expected_outcome_paths() -> tuple[set[str], set[str]]:
    directories = {
        _OUTCOME_CASE_ROOT,
        f"{_OUTCOME_CASE_ROOT}/staged-aligned",
        _OUTCOME_EPISODE_ROOT,
        f"{_OUTCOME_EPISODE_ROOT}/pcd_clean",
        f"{_OUTCOME_EPISODE_ROOT}/robot",
        f"{_OUTCOME_EPISODE_ROOT}/splatfacto",
    }
    files = {
        f"{_OUTCOME_CASE_ROOT}/held_outcome.json",
        f"{_OUTCOME_CASE_ROOT}/official_target.npz",
        f"{_OUTCOME_EPISODE_ROOT}/extrinsics.npy",
        f"{_OUTCOME_EPISODE_ROOT}/undistorted_intrinsics.npy",
        f"{_OUTCOME_EPISODE_ROOT}/pcd_clean/pcd_clean.meta.json",
        f"{_OUTCOME_EPISODE_ROOT}/robot/robot.meta.json",
        f"{_OUTCOME_EPISODE_ROOT}/robot/robot.npz",
        f"{_OUTCOME_EPISODE_ROOT}/splatfacto/splatfacto.meta.json",
    }
    for camera in _STAGED_CAMERAS:
        camera_root = f"{_OUTCOME_EPISODE_ROOT}/{camera}"
        tracking_root = f"{camera_root}/tracking"
        directories.update({camera_root, tracking_root})
        files.update(
            {
                f"{camera_root}/aligned_timestamps.txt",
                f"{camera_root}/mask_refined.h5",
                f"{camera_root}/metadata.json",
                f"{camera_root}/rendered_depth.h5",
                f"{camera_root}/rendered_depth.meta.json",
                f"{camera_root}/undistorted.mp4",
                f"{tracking_root}/tracking.meta.json",
                f"{tracking_root}/vel.h5",
                f"{tracking_root}/visibility.h5",
            }
        )
    files.update(
        f"{_OUTCOME_EPISODE_ROOT}/pcd_clean/{frame:06d}.npz" for frame in range(76)
    )
    files.update(
        f"{_OUTCOME_EPISODE_ROOT}/splatfacto/splat_{frame}.ply" for frame in range(81)
    )
    return directories, files


def _expected_paths() -> tuple[set[str], set[str]]:
    directories = {
        "calibration",
        "calibration/.outcome-phase.claim",
        "calibration/.shard-0.claim",
        "calibration/.shard-1.claim",
        "calibration/cases",
        "calibration/logs",
        "calibration/outcomes",
    }
    files = set(_EXPECTED_ROOT_FILES)
    files.update(
        f"calibration/logs/{case}.{suffix}"
        for case in _CALIBRATION_CASES
        for suffix in _CASE_LOG_SUFFIXES
    )
    files.update(
        {
            "calibration/logs/shard-0.lock-verification.log",
            "calibration/logs/shard-1.lock-verification.log",
        }
    )
    for case in _CALIBRATION_CASES:
        case_root = f"calibration/cases/{case}"
        directories.update(
            case_root if not suffix else f"{case_root}/{suffix}"
            for suffix in _CASE_DIRECTORY_SUFFIXES
        )
        files.update(f"{case_root}/{suffix}" for suffix in _CASE_FILE_SUFFIXES)
    outcome_directories, outcome_files = _expected_outcome_paths()
    directories.update(outcome_directories)
    files.update(outcome_files)
    return directories, files


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    files = [row for row in rows if row["type"] == "file"]
    directories = [row for row in rows if row["type"] == "directory"]
    return {
        "directory_count": len(directories),
        "file_count": len(files),
        "inventory_entry_count": len(rows),
        "inventory_sha256": hashlib.sha256(_canonical_inventory(rows)).hexdigest(),
        "total_file_bytes": sum(int(row["size"]) for row in files),
    }


def _category_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, expected in _EXPECTED_CATEGORIES.items():
        prefix = str(expected["path_prefix"])
        selected = [row for row in rows if str(row["path"]).startswith(prefix)]
        summaries[name] = {"path_prefix": prefix, **_summary(selected)}
    return summaries


def _inventory() -> list[dict[str, Any]]:
    _require(_HELD_ROOT.is_dir() and not _HELD_ROOT.is_symlink(), "bad held-v7 root")
    _require(_HELD_ROOT.resolve() == _HELD_ROOT, "non-canonical held-v7 root")
    root_names = {entry.name for entry in os.scandir(_HELD_ROOT)}
    if os.path.lexists(_REPORT):
        _require(
            _REPORT.is_file() and not _REPORT.is_symlink(),
            "v7 withdrawal report is not a regular non-symlink file",
        )
        root_names.remove(_REPORT.name)
    _require(
        root_names == {"calibration", _CODE_NAME, *_EXPECTED_ROOT_FILES},
        "unexpected held-v7 root inventory",
    )
    code = _HELD_ROOT / _CODE_NAME
    _require(
        code.is_dir() and not code.is_symlink() and code.resolve() == code,
        "bad held-v7 deployed-code directory",
    )
    _require(
        code.lstat().st_mode & 0o222 == 0,
        "held-v7 deployed-code root is writable",
    )

    rows: list[dict[str, Any]] = []
    for parent, directories, files in os.walk(_HELD_ROOT, followlinks=False):
        parent_path = Path(parent)
        relative_parent = parent_path.relative_to(_HELD_ROOT)
        if relative_parent == Path("."):
            directories[:] = [name for name in directories if name != _CODE_NAME]
        for name in sorted(directories):
            path = parent_path / name
            _require(not path.is_symlink(), f"directory symlink refused: {path}")
            rows.append(
                {
                    "path": path.relative_to(_HELD_ROOT).as_posix(),
                    "type": "directory",
                }
            )
        for name in sorted(files):
            path = parent_path / name
            if path == _REPORT:
                continue
            size, sha256 = _sha256_regular_file(path)
            rows.append(
                {
                    "path": path.relative_to(_HELD_ROOT).as_posix(),
                    "sha256": sha256,
                    "size": size,
                    "type": "file",
                }
            )
    rows.sort(key=lambda row: str(row["path"]))

    expected_directories, expected_files = _expected_paths()
    observed_directories = {
        str(row["path"]) for row in rows if row["type"] == "directory"
    }
    observed_files = {str(row["path"]) for row in rows if row["type"] == "file"}
    _require(
        observed_directories == expected_directories,
        "unexpected held-v7 evidence directory inventory",
    )
    _require(
        observed_files == expected_files,
        "unexpected held-v7 evidence file inventory",
    )
    _require(_summary(rows) == _EXPECTED_EVIDENCE, "held-v7 evidence census changed")
    _require(
        _category_summaries(rows) == _EXPECTED_CATEGORIES,
        "held-v7 category inventory changed",
    )

    indexed = {str(row["path"]): row for row in rows if row["type"] == "file"}
    for relative, (expected_size, expected_sha256) in {
        **_EXPECTED_ROOT_FILES,
        **_EXPECTED_KEY_OUTCOME_FILES,
    }.items():
        _require(
            indexed[relative]
            == {
                "path": relative,
                "sha256": expected_sha256,
                "size": expected_size,
                "type": "file",
            },
            f"held-v7 frozen evidence changed: {relative}",
        )
    return rows


def _file_records(values: Mapping[str, tuple[int, str]]) -> list[dict[str, Any]]:
    return [
        {"path": path, "sha256": sha256, "size": size}
        for path, (size, sha256) in sorted(values.items())
    ]


def expected_unsigned_report() -> dict[str, Any]:
    return {
        "artifact_kind": "Deform360HeldProtocolExecutionWithdrawalReport",
        "cause": {
            "cardinality_relation_disclosed_by_terminal_failure": (
                "eligible visible-and-valid official frame-zero identity count "
                "is less than sealed frame-zero point count"
            ),
            "classification": (
                "INSUFFICIENT_VISIBLE_VALID_OFFICIAL_FRAME_ZERO_IDENTITIES"
            ),
            "exception_message": (
                "too few visible and valid official frame-zero identities"
            ),
            "exception_type": "ValueError",
            "failed_case": _FAILED_CASE,
            "failure_phase": (
                "first calibration case identity-transport eligibility-cardinality "
                "precondition after completed CREATE target operation and before "
                "sparse assignment or metric computation"
            ),
            "terminal_exit_code": 2,
            "terminal_log": _file_records(
                {
                    "calibration-outcomes.console.log": _EXPECTED_ROOT_FILES[
                        "calibration-outcomes.console.log"
                    ]
                }
            )[0],
        },
        "deployed_method": {
            "git_head": "a5c4bcb463e3eaba066280b811d25382789c1104",
            "git_commit_object_sha256": (
                "6cd3e2b7568a7387f233b00ab412504fbc2c3224bbc8d6c262c41d16f67cfbe9"
            ),
            "git_tree_manifest_sha256": (
                "a142ebb294e7cf8ac71fe54f0071346dd3dc80d1c9766dce0319040fbf1c36ee"
            ),
            "snapshot_tree_sha256": (
                "74656761d73e4e010ff00aecfe1a49dc5f6ab99b8c99a5090c3ec3647e82ab96"
            ),
            "tracked_file_count": 901,
        },
        "disposition": (
            "WITHDRAWN_AFTER_FIRST_COMPLETED_TARGET_BEFORE_ANY_COMPLETED_CASE_SCORE"
        ),
        "evidence": {
            "canonical_held_root": os.fspath(_HELD_ROOT),
            "category_inventories": _EXPECTED_CATEGORIES,
            "complete_noncode_inventory": _EXPECTED_EVIDENCE,
            "inventory_canonicalization": (
                "path-sorted JSON rows with path/type and, for regular files, "
                "size/SHA-256; compact sorted-key JSON plus one newline"
            ),
            "inventory_scope": (
                "complete pre-report non-code execution evidence, including the "
                "completed first official target and all reconstruction staging; "
                "excludes the separately bound deployed code tree and this report"
            ),
            "key_completed_outcome_file_inventory": _file_records(
                _EXPECTED_KEY_OUTCOME_FILES
            ),
            "root_file_inventory": _file_records(_EXPECTED_ROOT_FILES),
            "stable_full_inventory_pass_count_before_sealing": 3,
            "structured_terminal_event_counts": {
                "calibration_cohort_barrier_validated": 1,
                "calibration_gate_decision_written": 0,
                "calibration_score_evidence_written": 0,
                "calibration_target_operation_complete": 1,
                "calibration_target_operation_planned": 15,
                "calibration_target_operation_start": 1,
                "fail_closed": 1,
                "gsplat_runtime_smoke_validated": 1,
            },
        },
        "execution_counts": {
            "calibration_case_execution_count": 15,
            "calibration_case_score_completed_count": 0,
            "calibration_decision_count": 0,
            "calibration_lock_count": 1,
            "calibration_score_evidence_count": 0,
            "confirmation_case_execution_count": 0,
            "confirmation_lock_count": 0,
            "confirmation_prediction_seal_count": 0,
            "completed_target_staging_directory_count": 21,
            "completed_target_staging_file_count": 235,
            "deployed_snapshot_count": 1,
            "formal_online_prediction_count": 15,
            "formal_physical_prediction_count": 15,
            "formal_outcome_runtime_smoke_validated_count": 1,
            "frame_zero_bundle_count": 15,
            "frame_zero_manifest_count": 15,
            "held_outcome_manifest_count": 1,
            "identity_transport_attempted_count": 1,
            "identity_transport_completed_count": 0,
            "metric_computation_started_count": 0,
            "online_prediction_seal_count": 15,
            "outcome_created_count": 1,
            "outcome_permit_count": 1,
            "outcome_phase_claim_count": 1,
            "outcome_read_count": 0,
            "pcd_clean_frame_archive_count": 76,
            "physical_prior_seal_count": 15,
            "prefix_authorization_count": 15,
            "rendered_depth_archive_count": 8,
            "sam2_camera_propagation_completed_count": 8,
            "sam2_frame_count_per_camera": 81,
            "sam2_mask_archive_count": 8,
            "shard_start_count": 2,
            "sparse_identity_assignment_started_count": 0,
            "splatfacto_ply_count": 81,
            "staged_camera_video_count": 8,
            "target_operation_completed_count": 1,
            "target_operation_failed_count": 0,
            "target_operation_planned_count": 15,
            "target_operation_started_count": 1,
            "target_reconstruction_artifact_count": 1,
            "target_reconstruction_completed_count": 1,
            "target_reconstruction_training_completed_count": 1,
            "target_reconstruction_training_started_count": 1,
            "tracking_velocity_archive_count": 8,
            "tracking_visibility_archive_count": 8,
        },
        "information_boundary": {
            "all_15_calibration_predictions_exist_and_are_sealed": True,
            "all_15_prediction_artifact_sets_revalidated_bytewise_for_outcome_permit": True,
            "calibration_gate_or_metric_created_or_read": False,
            "confirmation_payload_read": False,
            "exact_identity_cardinalities_in_withdrawal_report": False,
            "first_case_identity_eligibility_relation_evaluated": True,
            "first_case_official_target_arrays_constructed": True,
            "first_case_online_prediction_arrays_decoded_before_target_callback": True,
            "forensic_audit_disclosed_arrays_images_masks_metrics_or_protected_values": False,
            "forensic_audit_method": (
                "filenames/stat metadata and stable O_NOFOLLOW SHA-256 byte streams "
                "only; no payload was deserialized and no image or video was decoded"
            ),
            "future_tactile_read": False,
            "later_case_online_prediction_arrays_decoded": False,
            "object_future_mask_archive_created": (
                "CONFIRMED_WITHIN_FIRST_CALIBRATION_CASE_ONLY"
            ),
            "object_future_mask_archive_count_upper_bound": 8,
            "object_future_mask_downstream_read": True,
            "object_future_rgb_read": "CONFIRMED_WITHIN_FIRST_CALIBRATION_CASE_ONLY",
            "object_future_rgb_read_case_upper_bound": 1,
            "rendered_future_depth_archive_count": 8,
            "rendered_future_depth_downstream_read": True,
            "source_dataset_future_depth_read": False,
            "source_dataset_future_tracking_read": False,
            "derived_future_tracking_downstream_read": True,
            "derived_future_tracking_velocity_archive_count": 8,
            "derived_future_tracking_visibility_archive_count": 8,
            "official_target_reconstruction_created": True,
            "official_target_reconstruction_count_upper_bound": 1,
            "partial_or_completed_target_source_staging_scope": (
                "one calibration case; eight camera videos, masks, rendered-depth "
                "and tracking archives; 76 cleaned point-cloud frames; 81 Splatfacto "
                "PLY frames; camera calibration; robot metadata/archive; one official "
                "target archive and one held-outcome manifest"
            ),
            "protected_cardinality_values_returned_by_forensic_audit": False,
            "sparse_identity_assignment_created": False,
            "target_arrays_coordinates_metrics_or_labels_returned_by_forensic_audit": False,
            "tactile_read": False,
        },
        "protocol_id": _PROTOCOL_ID,
        "replacement_protocol_id": _REPLACEMENT_PROTOCOL_ID,
        "result_status": "NO_CALIBRATION_RESULT",
        "reuse": {
            "v7_completed_target_or_staging_reused_by_v8": False,
            "v7_evidence_may_be_used_by_v8_only_as_immutable_lineage": True,
            "v7_execution_artifacts_reused_by_v8": False,
            "v7_physical_or_online_predictions_reused_by_v8": False,
            "v7_score_or_gate_available_for_reuse": False,
            "v8_requires_fresh_absent_held_root": True,
            "v8_requires_fresh_predictions_and_outcome_phase": True,
        },
        "schema_version": 1,
    }


def _write_once(payload: bytes) -> None:
    descriptor = os.open(
        _REPORT,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(_REPORT, 0o400, follow_symlinks=False)


def _seal_permissions(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if row["type"] == "file":
            os.chmod(_HELD_ROOT / str(row["path"]), 0o400, follow_symlinks=False)
    os.chmod(_REPORT, 0o400, follow_symlinks=False)
    directories = [str(row["path"]) for row in rows if row["type"] == "directory"]
    for relative in sorted(
        directories, key=lambda value: value.count("/"), reverse=True
    ):
        os.chmod(_HELD_ROOT / relative, 0o500, follow_symlinks=False)
    os.chmod(_HELD_ROOT, 0o500, follow_symlinks=False)


def main() -> None:
    _require(
        socket.gethostname() == "workstation2",
        "v7 evidence may only be sealed on gpuserver6000/workstation2",
    )
    first = _inventory()
    second = _inventory()
    _require(first == second, "held-v7 evidence changed across pre-seal hash passes")
    signed, payload = _artifact(expected_unsigned_report())
    if os.path.lexists(_REPORT):
        size, digest = _sha256_regular_file(_REPORT)
        _require(size == len(payload), "existing v7 report length changed")
        _require(
            digest == hashlib.sha256(payload).hexdigest(),
            "existing v7 report changed",
        )
    else:
        _write_once(payload)
    third = _inventory()
    _require(first == third, "held-v7 evidence changed while report was written")
    _seal_permissions(third)
    _require(
        stat.S_IMODE(_REPORT.stat().st_mode) == 0o400,
        "v7 withdrawal report is not mode 0400",
    )
    print(
        json.dumps(
            {
                "artifact_sha256": signed["artifact_sha256"],
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "path": os.fspath(_REPORT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
