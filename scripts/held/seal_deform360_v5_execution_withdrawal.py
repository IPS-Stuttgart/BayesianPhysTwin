#!/usr/bin/env python3
"""Seal the failed held-v5 outcome launch and write its withdrawal report.

This is deliberately a one-purpose forensic operator. It inventories the
complete non-code held-v5 execution tree using only filenames, file metadata,
and SHA-256 byte streams. It never deserializes JSON/NPZ/NPY files, decodes a
video, reads a metric, or traverses the deployed code tree. The exact frozen
inventory must match before a report can be created.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


_HELD_ROOT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v5")
_REPORT = _HELD_ROOT / "v5-outcome-withdrawal-report.json"
_CODE_NAME = "code-db94e490c299fe97ca986be2d81a65be95444dcf"
_PROTOCOL_ID = "deform360-held-online-belief-v5"
_REPLACEMENT_PROTOCOL_ID = "deform360-held-online-belief-v6"

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
_FAILED_CAMERA = "brics-odroid-001_cam0"
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
        16_956,
        "a917650499b047bdcd7d7baf57212ff82a9e277867bb3ba5389b1a0c126d950e",
    ),
    "calibration-outcomes.console.log": (
        3_270,
        "7f48d2ee1291d37f051a9422e2217fafe7f08d7438285b8b4f4ee14e5d93ab71",
    ),
    "calibration-shard-0.console.log": (
        2_723,
        "084c4f85d30f111d54790471cacc6949e2f860b1612c8ad40b8cdf5975d15bf0",
    ),
    "calibration-shard-1.console.log": (
        2_383,
        "7b8a1f78dfb4a66ab84522e5c2fa74d1b1b4b600860b6e9877293168ef0cf0c1",
    ),
}

_OUTCOME_PREFIX = (
    "calibration/outcomes/002-rope-silk-ep0003/staged-aligned/episode_0000"
)
_EXPECTED_PARTIAL_OUTCOME_FILES: Mapping[str, tuple[int, str]] = {
    f"{_OUTCOME_PREFIX}/brics-odroid-001_cam0/aligned_timestamps.txt": (
        2_916,
        "5ae3f153df36438cacb117e322a93bb0c76f0b39e4b92cfe02d0a667057632f0",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-001_cam0/metadata.json": (
        1_942,
        "2ef7b079c8a101f74e52d0778ebd6bdc6520e6c7edbf546429a5bc20551ba8e5",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-001_cam0/undistorted.mp4": (
        26_065_640,
        "1b49534cb630cf8005b0d916a429d79cdf999bab68b2a4b30e1a1e364cc1b7a2",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-006_cam0/aligned_timestamps.txt": (
        2_916,
        "5ae3f153df36438cacb117e322a93bb0c76f0b39e4b92cfe02d0a667057632f0",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-006_cam0/metadata.json": (
        1_942,
        "a96bbe34a41acea5b4448d1bc5d4f211e6425daa4c8d4e69d2018343223031b5",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-006_cam0/undistorted.mp4": (
        18_824_877,
        "914f8422ba7fdd6f79fd66ed07882a72c130cfdacaee79e190aeb1a5c8e96737",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-007_cam0/aligned_timestamps.txt": (
        2_916,
        "5ae3f153df36438cacb117e322a93bb0c76f0b39e4b92cfe02d0a667057632f0",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-007_cam0/metadata.json": (
        1_942,
        "3f05519d75107dae1fd3db5b7badeed163aa53a4385e21e5d5225a8503891baf",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-007_cam0/undistorted.mp4": (
        21_094_762,
        "2e3dcc06eb62f09a50125d4f913b89600e9365ec7a7f359d94f68e7a404483d7",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-014_cam1/aligned_timestamps.txt": (
        2_916,
        "5ae3f153df36438cacb117e322a93bb0c76f0b39e4b92cfe02d0a667057632f0",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-014_cam1/metadata.json": (
        1_942,
        "e292c74bb48248db3faeb8f0b15fcce3826a35ceab47ad3a4a107dcfb63963c2",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-014_cam1/undistorted.mp4": (
        23_672_441,
        "3a3dd4b7c7c75790eacbac79ff1aa4895fe36609e7d4dd7a6963f75408757b27",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-021_cam1/aligned_timestamps.txt": (
        2_916,
        "5ae3f153df36438cacb117e322a93bb0c76f0b39e4b92cfe02d0a667057632f0",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-021_cam1/metadata.json": (
        1_942,
        "8869702416a6421d09c5609d42d59a82b278afabcd87b711323305daa3ab4039",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-021_cam1/undistorted.mp4": (
        22_171_693,
        "d67f22b4a5c21180a694797171e05f5028423457d5e55e139292b6d2674e2583",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-025_cam1/aligned_timestamps.txt": (
        2_916,
        "5ae3f153df36438cacb117e322a93bb0c76f0b39e4b92cfe02d0a667057632f0",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-025_cam1/metadata.json": (
        1_942,
        "2ef79b178e1681cfcab9ec3f510dd02d4fec6bbbc0ee403ca98ed525c07e8fdf",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-025_cam1/undistorted.mp4": (
        22_854_471,
        "83ef66a9ebd7aa3ac603e8df39d5eb6aa3621007732ef179e4b1d2e63242fdd4",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-027_cam0/aligned_timestamps.txt": (
        2_916,
        "5ae3f153df36438cacb117e322a93bb0c76f0b39e4b92cfe02d0a667057632f0",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-027_cam0/metadata.json": (
        1_942,
        "98e083fc73d7014a120c07f77b7b64ede2185bb8381415a23061d159b775f64f",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-027_cam0/undistorted.mp4": (
        18_930_215,
        "db147f4dcb681f64fbc0f28afb0d588e0162050bec93fbc2bf22dfe33e6a11dc",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-028_cam0/aligned_timestamps.txt": (
        2_916,
        "5ae3f153df36438cacb117e322a93bb0c76f0b39e4b92cfe02d0a667057632f0",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-028_cam0/metadata.json": (
        1_942,
        "c18d6dcd042cde182e27a1e779314dcdb5daf1936e0475c40dd8e36d0e83281e",
    ),
    f"{_OUTCOME_PREFIX}/brics-odroid-028_cam0/undistorted.mp4": (
        22_382_803,
        "12a8bc093b1a3c8fda03d39ea40832c0af4978cce232cca510f4858266d27100",
    ),
    f"{_OUTCOME_PREFIX}/extrinsics.npy": (
        1_881,
        "62b09e18a5ab91d6b9efb4e5e14e73325c2385464813443bd60db4ec9738590c",
    ),
    f"{_OUTCOME_PREFIX}/robot/robot.meta.json": (
        2_297,
        "0052d60adb8d5b258760f00d1b1036e36436c10723ffea217c19aae2696f85f0",
    ),
    f"{_OUTCOME_PREFIX}/robot/robot.npz": (
        17_211,
        "7090543c95200b9f07604cf6ece6f237a49e0181ee578dc9169727a205fb3fc8",
    ),
    f"{_OUTCOME_PREFIX}/undistorted_intrinsics.npy": (
        1_433,
        "bd4390fe93b2301939ba92c964757a13cb8c3204d349b866a3d6be7a9983dbd8",
    ),
}

_EXPECTED_EVIDENCE = {
    "directory_count": 124,
    "file_count": 574,
    "inventory_entry_count": 698,
    "inventory_sha256": (
        "b7684082abe5c778969f246959d858db613c9f9a1469609b2e962dc16e23043e"
    ),
    "total_file_bytes": 826_563_657,
}
_EXPECTED_CATEGORIES = {
    "case_artifacts": {
        "directory_count": 105,
        "file_count": 420,
        "inventory_entry_count": 525,
        "inventory_sha256": (
            "b14a4f24acf515c702672f5853304fefa3b0bf674273b443c2804d3c52c753e0"
        ),
        "path_prefix": "calibration/cases/",
        "total_file_bytes": 643_617_477,
    },
    "execution_logs": {
        "directory_count": 0,
        "file_count": 122,
        "inventory_entry_count": 122,
        "inventory_sha256": (
            "83d8472f9a6cf81361e5f078d70d3c8b60c8cd66c751c43a7a57f449467581a6"
        ),
        "path_prefix": "calibration/logs/",
        "total_file_bytes": 6_862_260,
    },
    "partial_outcome_staging": {
        "directory_count": 12,
        "file_count": 28,
        "inventory_entry_count": 40,
        "inventory_sha256": (
            "a1b9308754f960cf17f7a13531fe4fa1f4ca8577ca8d105a05ee63658082f4f6"
        ),
        "path_prefix": "calibration/outcomes/",
        "total_file_bytes": 176_058_588,
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


def _expected_paths() -> tuple[set[str], set[str]]:
    directories = {
        "calibration",
        "calibration/.outcome-phase.claim",
        "calibration/.shard-0.claim",
        "calibration/.shard-1.claim",
        "calibration/cases",
        "calibration/logs",
        "calibration/outcomes",
        f"calibration/outcomes/{_FAILED_CASE}",
        f"calibration/outcomes/{_FAILED_CASE}/staged-aligned",
        _OUTCOME_PREFIX,
        f"{_OUTCOME_PREFIX}/robot",
    }
    directories.update(f"{_OUTCOME_PREFIX}/{camera}" for camera in _STAGED_CAMERAS)
    files = set(_EXPECTED_ROOT_FILES) | set(_EXPECTED_PARTIAL_OUTCOME_FILES)
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
    _require(_HELD_ROOT.is_dir() and not _HELD_ROOT.is_symlink(), "bad held-v5 root")
    _require(_HELD_ROOT.resolve() == _HELD_ROOT, "non-canonical held-v5 root")
    root_names = {entry.name for entry in os.scandir(_HELD_ROOT)}
    if os.path.lexists(_REPORT):
        _require(
            _REPORT.is_file() and not _REPORT.is_symlink(),
            "v5 withdrawal report is not a regular non-symlink file",
        )
        root_names.remove(_REPORT.name)
    _require(
        root_names
        == {
            "calibration",
            "calibration-lock.json",
            "calibration-outcomes.console.log",
            "calibration-shard-0.console.log",
            "calibration-shard-1.console.log",
            _CODE_NAME,
        },
        "unexpected held-v5 root inventory",
    )
    code = _HELD_ROOT / _CODE_NAME
    _require(
        code.is_dir() and not code.is_symlink() and code.resolve() == code,
        "bad held-v5 deployed-code directory",
    )
    _require(
        code.lstat().st_mode & 0o222 == 0,
        "held-v5 deployed-code root is writable",
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
        "unexpected held-v5 evidence directory inventory",
    )
    _require(
        observed_files == expected_files,
        "unexpected held-v5 evidence file inventory",
    )
    _require(_summary(rows) == _EXPECTED_EVIDENCE, "held-v5 evidence census changed")
    _require(
        _category_summaries(rows) == _EXPECTED_CATEGORIES,
        "held-v5 category inventory changed",
    )

    indexed = {str(row["path"]): row for row in rows if row["type"] == "file"}
    for relative, (expected_size, expected_sha256) in {
        **_EXPECTED_ROOT_FILES,
        **_EXPECTED_PARTIAL_OUTCOME_FILES,
    }.items():
        _require(
            indexed[relative]
            == {
                "path": relative,
                "sha256": expected_sha256,
                "size": expected_size,
                "type": "file",
            },
            f"held-v5 frozen evidence changed: {relative}",
        )
    return rows


def _file_records(
    values: Mapping[str, tuple[int, str]],
) -> list[dict[str, Any]]:
    return [
        {"path": path, "sha256": sha256, "size": size}
        for path, (size, sha256) in sorted(values.items())
    ]


def expected_unsigned_report() -> dict[str, Any]:
    return {
        "artifact_kind": "Deform360HeldProtocolExecutionWithdrawalReport",
        "cause": {
            "classification": "PROPAGATED_FRAME_ZERO_MASK_SEAL_MISMATCH",
            "exception_message": (
                "propagated frame-zero mask differs from seal: " + _FAILED_CAMERA
            ),
            "failed_camera": _FAILED_CAMERA,
            "failed_case": _FAILED_CASE,
            "failure_phase": "first calibration target reconstruction operation",
            "terminal_log": _file_records(
                {
                    "calibration-outcomes.console.log": _EXPECTED_ROOT_FILES[
                        "calibration-outcomes.console.log"
                    ]
                }
            )[0],
        },
        "deployed_method": {
            "git_head": "db94e490c299fe97ca986be2d81a65be95444dcf",
            "git_commit_object_sha256": (
                "d2b08f3a2247c02e98655dfc2f69eb03d1c72c3c551da04bd8e01a7841069b0b"
            ),
            "git_tree_manifest_sha256": (
                "4a0f546c79dee06f7c93c6254eb7698396a70d5fe38a197373d71d898ce3ab83"
            ),
            "snapshot_tree_sha256": (
                "ec255c6c413119da6c8a12864ae8f03fc2ccc6249744840b9b85abf320900586"
            ),
            "tracked_file_count": 878,
        },
        "disposition": (
            "WITHDRAWN_DURING_FIRST_TARGET_OPERATION_BEFORE_ANY_COMPLETED_OUTCOME"
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
                "complete pre-report non-code execution evidence, including partial "
                "target staging; excludes the separately bound deployed code tree "
                "and this withdrawal report"
            ),
            "partial_outcome_file_inventory": _file_records(
                _EXPECTED_PARTIAL_OUTCOME_FILES
            ),
            "root_file_inventory": _file_records(_EXPECTED_ROOT_FILES),
            "stable_full_inventory_pass_count_before_sealing": 3,
        },
        "execution_counts": {
            "calibration_case_execution_count": 15,
            "calibration_decision_count": 0,
            "calibration_lock_count": 1,
            "calibration_score_evidence_count": 0,
            "confirmation_case_execution_count": 0,
            "confirmation_lock_count": 0,
            "confirmation_prediction_seal_count": 0,
            "deployed_snapshot_count": 1,
            "formal_online_prediction_count": 15,
            "formal_physical_prediction_count": 15,
            "frame_zero_bundle_count": 15,
            "frame_zero_manifest_count": 15,
            "online_prediction_seal_count": 15,
            "outcome_created_count": 0,
            "outcome_permit_count": 1,
            "outcome_phase_claim_count": 1,
            "outcome_read_count": 0,
            "partial_target_case_directory_count": 1,
            "partial_target_staging_directory_count": 12,
            "partial_target_staging_file_count": 28,
            "physical_prior_seal_count": 15,
            "prefix_authorization_count": 15,
            "shard_start_count": 2,
            "staged_camera_video_count": 8,
            "target_operation_completed_count": 0,
            "target_operation_failed_count": 1,
            "target_operation_planned_count": 15,
            "target_operation_started_count": 1,
            "target_reconstruction_artifact_count": 0,
        },
        "information_boundary": {
            "all_15_calibration_predictions_exist_and_are_sealed": True,
            "all_15_prediction_artifact_sets_revalidated_bytewise_for_outcome_permit": True,
            "calibration_gate_or_metric_created_or_read": False,
            "confirmation_payload_read": False,
            "first_case_online_prediction_arrays_decoded_before_target_callback": True,
            "forensic_audit_disclosed_arrays_images_masks_metrics_or_protected_values": False,
            "forensic_audit_method": (
                "filenames/stat metadata and stable O_NOFOLLOW SHA-256 byte streams "
                "only; no file was deserialized and no video was decoded"
            ),
            "future_tactile_read": False,
            "later_case_online_prediction_arrays_decoded": False,
            "object_future_depth_read": False,
            "object_future_rgb_read": "POSSIBLE_WITHIN_FIRST_CALIBRATION_CASE_ONLY",
            "object_future_rgb_read_case_upper_bound": 1,
            "object_future_rgb_read_reason": (
                "the first target callback started and eight staged video files "
                "exist; the metadata-only audit cannot establish the exact source "
                "frames decoded before failure"
            ),
            "object_future_tracking_read": False,
            "official_target_reconstruction_created": False,
            "partial_target_source_staging_created": True,
            "partial_target_source_staging_scope": (
                "one calibration case, eight camera videos and timestamps/metadata, "
                "camera calibration, and robot metadata/archive"
            ),
            "tactile_read": False,
        },
        "protocol_id": _PROTOCOL_ID,
        "replacement_protocol_id": _REPLACEMENT_PROTOCOL_ID,
        "reuse": {
            "v5_evidence_may_be_used_by_v6_only_as_immutable_lineage": True,
            "v5_execution_artifacts_reused_by_v6": False,
            "v5_partial_target_staging_reused_by_v6": False,
            "v5_physical_or_online_predictions_reused_by_v6": False,
            "v5_score_or_gate_available_for_reuse": False,
            "v6_requires_fresh_absent_held_root": True,
            "v6_requires_fresh_predictions_and_outcome_phase": True,
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
        "v5 evidence may only be sealed on gpuserver6000/workstation2",
    )
    first = _inventory()
    second = _inventory()
    _require(first == second, "held-v5 evidence changed across pre-seal hash passes")
    signed, payload = _artifact(expected_unsigned_report())
    if os.path.lexists(_REPORT):
        size, digest = _sha256_regular_file(_REPORT)
        _require(size == len(payload), "existing v5 report length changed")
        _require(
            digest == hashlib.sha256(payload).hexdigest(),
            "existing v5 report changed",
        )
    else:
        _write_once(payload)
    third = _inventory()
    _require(first == third, "held-v5 evidence changed while report was written")
    _seal_permissions(third)
    _require(
        stat.S_IMODE(_REPORT.stat().st_mode) == 0o400,
        "v5 withdrawal report is not mode 0400",
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
