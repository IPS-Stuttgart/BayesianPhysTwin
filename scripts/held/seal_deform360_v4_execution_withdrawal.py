#!/usr/bin/env python3
"""Seal the failed held-v4 launch and write its exact withdrawal report.

This is deliberately a one-purpose forensic operator.  It only opens the
already-public lock/lineage files, the two source-only frame-zero bundles, and
the launch diagnostics named in :data:`_EXPECTED_FILES`.  It never traverses
the Deform360 dataset and it refuses any unexpected entry below ``calibration``
before opening an execution artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from pathlib import Path
from typing import Any, Mapping


_HELD_ROOT = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1/held-v4")
_REPORT = _HELD_ROOT / "v4-execution-withdrawal-report.json"
_PROTOCOL_ID = "deform360-held-online-belief-v4"
_REPLACEMENT_PROTOCOL_ID = "deform360-held-online-belief-v5"
_LOCKED_PIP_FREEZE_SHA256 = (
    "e573fcaaa5f5006fb380bddd1d258fadafe9f6bddfc7838faec363841832ecd5"
)
_POST_FAILURE_PIP_FREEZE_SHA256 = (
    "4948737892f77c6a9496795e6c3f25b92fcea466ddb7b5f1e9c1b0de1137f004"
)

_EXPECTED_DIRECTORIES = frozenset(
    {
        "calibration/.shard-0.claim",
        "calibration/.shard-1.claim",
        "calibration/cases",
        "calibration/cases/002-rope-silk-ep0003",
        "calibration/cases/002-rope-silk-ep0003/frame-zero",
        "calibration/cases/083-blanket-cloth-ep0003",
        "calibration/cases/083-blanket-cloth-ep0003/frame-zero",
        "calibration/logs",
    }
)

# Relative path -> (byte length, SHA-256).  These are the complete non-code
# evidence inventory for the launch.  In particular, there is no physical,
# online, permit, or outcome artifact in the inventory.
_EXPECTED_FILES: Mapping[str, tuple[int, str]] = {
    "calibration-lock.json": (
        16_746,
        "3f5b6b678c095cd16e5aec1fdb8d0a6ad690e7a7e26c373b4740675e3399dacb",
    ),
    "calibration-shard-0.console.log": (
        488,
        "130c9d60807c42decd31e32faa7709aed65ed11d966f7a9a3f340c0b056ff408",
    ),
    "calibration-shard-1.console.log": (
        500,
        "38bc30660eb9cead9b456374839d5c9c3594529a84f433ae03b8935e995c8af9",
    ),
    "calibration/cases/002-rope-silk-ep0003/frame-zero/"
    "frame_zero_bundle.manifest.json": (
        384_583,
        "be1ba0a036c5fc6aac87af756b49cfbdc4a1026050c62241792283ab00e69583",
    ),
    "calibration/cases/002-rope-silk-ep0003/frame-zero/frame_zero_bundle.npz": (
        7_399_976,
        "fafa4a6df08839726da68b269c0be07c14cd6396dc5929e55917ff099ad442fa",
    ),
    "calibration/cases/002-rope-silk-ep0003/frame-zero/known_action_76.npz": (
        16_213,
        "5805d3394c4b46f5d7764e0cd6043a4b44416ac36951b4e3e96a01ec4a2e3edf",
    ),
    "calibration/cases/083-blanket-cloth-ep0003/frame-zero/"
    "frame_zero_bundle.manifest.json": (
        85_859,
        "13cf21cb4ec8c9b27a2bee0eadbffc3b5185b2b258722c952203ea355f631ea2",
    ),
    "calibration/cases/083-blanket-cloth-ep0003/frame-zero/frame_zero_bundle.npz": (
        13_985_776,
        "ef66563fa8c21030ab01f12fe1ef07e553407281fc2d52f522ad6efb94c12447",
    ),
    "calibration/cases/083-blanket-cloth-ep0003/frame-zero/known_action_76.npz": (
        16_362,
        "c9a174c8f63ab74d8e41ba9415adc5661499a895a94eef9cc663ad50edef0bff",
    ),
    "calibration/logs/002-rope-silk-ep0003.frame-zero-validate.log": (
        125,
        "52716c1e6e135694767d01b8efeaab6d8ef13d240f4464eb38f58059d31ab8f3",
    ),
    "calibration/logs/002-rope-silk-ep0003.frame-zero.log": (
        35_449,
        "f903ae79fb82b3efa7afad23c26741ac876dd73ce7daa53bb6c8f64c34c038df",
    ),
    "calibration/logs/002-rope-silk-ep0003.physical.failed.log": (
        1_498,
        "ffd851508b34664f26f238b81fca43ee3a4d22d211a8d90efd5e48f355ad9fae",
    ),
    "calibration/logs/083-blanket-cloth-ep0003.frame-zero-validate.log": (
        129,
        "77cfc6d5fb143b2e2028d29ed26d57fb20c703f61056fee96c23f1066977dbd7",
    ),
    "calibration/logs/083-blanket-cloth-ep0003.frame-zero.log": (
        11_043,
        "09925b0c411071ca484b14c446452477f44cc6b8a8cc9b22aa2008115368626d",
    ),
    "calibration/logs/083-blanket-cloth-ep0003.physical.failed.log": (
        1_498,
        "ffd851508b34664f26f238b81fca43ee3a4d22d211a8d90efd5e48f355ad9fae",
    ),
    "calibration/logs/shard-0.lock-verification.log": (
        970,
        "e06e73bbebcddf8aba3c8da3f0c55cd2e47f29537a526b855f7c4249b3cee552",
    ),
    "calibration/logs/shard-1.lock-verification.log": (
        970,
        "e06e73bbebcddf8aba3c8da3f0c55cd2e47f29537a526b855f7c4249b3cee552",
    ),
    "v2-design-withdrawal-report.json": (
        2_018,
        "a7cf04337dbdccc1e3e2165f89b7c51bb25b53bc3c89dc54ddbdf7b5df5dadb3",
    ),
    "v3-prelock-boundary-incident-report.json": (
        3_580,
        "b344a99cb6c4de4fe16b186f85f914dc7d2a3e049eac90b3ae40b56381c4505d",
    ),
}

_EXPECTED_ROOT_ENTRIES = frozenset(
    {
        "calibration",
        "calibration-lock.json",
        "calibration-shard-0.console.log",
        "calibration-shard-1.console.log",
        "code-dd5dce635b6a884ae86c9d179c3f5928d8004f3b",
        "v2-design-withdrawal-report.json",
        "v3-prelock-boundary-incident-report.json",
    }
)


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
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
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


def _inventory() -> list[dict[str, Any]]:
    _require(_HELD_ROOT.is_dir() and not _HELD_ROOT.is_symlink(), "bad held-v4 root")
    _require(_HELD_ROOT.resolve() == _HELD_ROOT, "non-canonical held-v4 root")
    root_names = {entry.name for entry in os.scandir(_HELD_ROOT)}
    if os.path.lexists(_REPORT):
        _require(
            _REPORT.is_file() and not _REPORT.is_symlink(),
            "v4 withdrawal report is not a regular non-symlink file",
        )
        root_names.remove(_REPORT.name)
    _require(root_names == _EXPECTED_ROOT_ENTRIES, "unexpected held-v4 root inventory")

    calibration = _HELD_ROOT / "calibration"
    _require(
        calibration.is_dir()
        and not calibration.is_symlink()
        and calibration.resolve() == calibration,
        "bad held-v4 calibration directory",
    )
    code = _HELD_ROOT / "code-dd5dce635b6a884ae86c9d179c3f5928d8004f3b"
    _require(
        code.is_dir() and not code.is_symlink() and code.resolve() == code,
        "bad held-v4 deployed-code directory",
    )
    observed_calibration: set[str] = set()
    for parent, directories, files in os.walk(calibration, followlinks=False):
        parent_path = Path(parent)
        for name in directories:
            path = parent_path / name
            _require(not path.is_symlink(), f"directory symlink refused: {path}")
            observed_calibration.add(path.relative_to(_HELD_ROOT).as_posix())
        for name in files:
            path = parent_path / name
            _require(not path.is_symlink(), f"file symlink refused: {path}")
            observed_calibration.add(path.relative_to(_HELD_ROOT).as_posix())
    expected_calibration = _EXPECTED_DIRECTORIES | {
        path for path in _EXPECTED_FILES if path.startswith("calibration/")
    }
    _require(
        observed_calibration == expected_calibration,
        "unexpected held-v4 calibration inventory",
    )

    rows: list[dict[str, Any]] = []
    for relative, (expected_size, expected_sha256) in sorted(_EXPECTED_FILES.items()):
        size, sha256 = _sha256_regular_file(_HELD_ROOT / relative)
        _require(size == expected_size, f"held-v4 evidence size changed: {relative}")
        _require(
            sha256 == expected_sha256,
            f"held-v4 evidence checksum changed: {relative}",
        )
        rows.append({"path": relative, "sha256": sha256, "size": size})
    return rows


def expected_unsigned_report() -> dict[str, Any]:
    inventory = [
        {"path": path, "sha256": sha256, "size": size}
        for path, (size, sha256) in sorted(_EXPECTED_FILES.items())
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()
    return {
        "artifact_kind": "Deform360HeldProtocolExecutionWithdrawalReport",
        "attempted_cases": [
            "002-rope-silk-ep0003",
            "083-blanket-cloth-ep0003",
        ],
        "cause": {
            "classification": "IMMUTABLE_PYTHON_INVENTORY_MISMATCH",
            "exception_message": "Python pip freeze differs from the immutable lock",
            "failure_phase": "physical-build runtime preflight",
            "failure_time_observed_pip_freeze_sha256": "NOT_RECORDED_BY_V4",
            "locked_pip_freeze_sorted_sha256": _LOCKED_PIP_FREEZE_SHA256,
            "post_failure_diagnostic": {
                "interpretation": (
                    "the post-failure mismatch is consistent with shared virtualenv "
                    "inventory drift, but is not claimed as a reconstruction of the "
                    "unrecorded failure-time inventory"
                ),
                "isolated_and_nonisolated_probe_hashes_equal": True,
                "observed_pip_freeze_sorted_sha256": (_POST_FAILURE_PIP_FREEZE_SHA256),
            },
        },
        "deployed_method": {
            "git_head": "dd5dce635b6a884ae86c9d179c3f5928d8004f3b",
            "git_commit_object_sha256": (
                "716e60758a46706e4bc48992f03c4cb617bdebef4da35d673e7cd4943ebb89a4"
            ),
            "git_tree_manifest_sha256": (
                "d25a61db4084953f9872d4b8857ff6e3c52ec8751d2f0d129fa0cc8f52794171"
            ),
            "snapshot_tree_sha256": (
                "c4c2971ef12b875f979d0a7385e7d35b8da5e94a3858bdebd0acf484a8d1e1e7"
            ),
        },
        "disposition": "WITHDRAWN_AFTER_FRAME_ZERO_BEFORE_PHYSICAL_PREDICTION",
        "evidence": {
            "canonical_held_root": os.fspath(_HELD_ROOT),
            "file_inventory": inventory,
            "file_inventory_scope": (
                "complete pre-report non-code execution evidence; excludes the "
                "separately hash-bound deployed method tree and this withdrawal report"
            ),
            "file_inventory_sha256": inventory_sha256,
        },
        "execution_counts": {
            "calibration_decision_count": 0,
            "calibration_lock_count": 1,
            "case_attempt_count": 2,
            "confirmation_lock_count": 0,
            "deployed_snapshot_count": 1,
            "deployment_count": 1,
            "frame_zero_bundle_count": 2,
            "frame_zero_manifest_count": 2,
            "formal_online_prediction_count": 0,
            "formal_physical_prediction_count": 0,
            "online_prediction_seal_count": 0,
            "outcome_api_operation_count": 0,
            "outcome_created_count": 0,
            "outcome_permit_count": 0,
            "outcome_read_count": 0,
            "physical_builder_invocation_count": 2,
            "physical_prediction_artifact_count": 0,
            "physical_prior_seal_count": 0,
            "prefix_authorization_count": 0,
            "shard_start_count": 2,
            "target_operation_count": 0,
        },
        "information_boundary": {
            "case_operator_frame_zero_build_and_validation_completed_before_physical_runtime_preflight": True,
            "confirmation_payload_read": False,
            "episode_payload_read": True,
            "episode_payload_read_scope": (
                "frame-zero RGB-D and masks; the frame-zero pipeline read the full "
                "realized robot archive to select the window, then sealed the aligned "
                "76-frame robot-kinematics window"
            ),
            "frame_zero_source_artifacts_created": True,
            "future_tactile_read": False,
            "object_future_depth_read": False,
            "object_future_rgb_read": False,
            "object_future_tracking_read": False,
            "online_prediction_created": False,
            "outcome_created_or_read": False,
            "physical_runtime_validation_failed_before_physical_output_directory_creation": True,
            "physical_prediction_created": False,
            "prediction_payload_read": False,
            "tactile_read": False,
            "target_data_read": False,
            "target_operation_executed": False,
        },
        "protocol_id": _PROTOCOL_ID,
        "replacement_protocol_id": _REPLACEMENT_PROTOCOL_ID,
        "reuse": {
            "v4_frame_zero_artifacts_reused_by_v5": False,
            "v4_physical_or_online_predictions_reused_by_v5": False,
            "v4_formal_physical_or_online_prediction_count_available_for_reuse": 0,
            "v5_requires_fresh_absent_held_root": True,
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


def main() -> None:
    _require(
        socket.gethostname() == "workstation2",
        "v4 evidence may only be sealed on gpuserver6000/workstation2",
    )
    observed_inventory = _inventory()
    expected = expected_unsigned_report()
    _require(
        observed_inventory == expected["evidence"]["file_inventory"],
        "held-v4 evidence inventory changed",
    )
    signed, payload = _artifact(expected)
    if os.path.lexists(_REPORT):
        size, digest = _sha256_regular_file(_REPORT)
        _require(size == len(payload), "existing v4 report length changed")
        _require(
            digest == hashlib.sha256(payload).hexdigest(),
            "existing v4 report changed",
        )
    else:
        _write_once(payload)

    # Make every execution-evidence leaf and directory owner-read-only only
    # after the exact report exists.  The deployed code was already sealed.
    for relative in _EXPECTED_FILES:
        os.chmod(_HELD_ROOT / relative, 0o400, follow_symlinks=False)
    os.chmod(_REPORT, 0o400, follow_symlinks=False)
    for relative in sorted(
        _EXPECTED_DIRECTORIES, key=lambda value: value.count("/"), reverse=True
    ):
        os.chmod(_HELD_ROOT / relative, 0o500, follow_symlinks=False)
    os.chmod(_HELD_ROOT / "calibration", 0o500, follow_symlinks=False)
    os.chmod(_HELD_ROOT, 0o500, follow_symlinks=False)
    _require(
        stat.S_IMODE(_REPORT.stat().st_mode) == 0o400,
        "v4 withdrawal report is not mode 0400",
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
