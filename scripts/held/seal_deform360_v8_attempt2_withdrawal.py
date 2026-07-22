#!/usr/bin/env python3
"""Seal and archive held-v8 attempt 2 before any outcome access."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import stat
from typing import Any, Mapping


BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
ACTIVE = BASE / "held-v8"
ARCHIVE = BASE / "held-v8-attempt-2-withdrawn-preoutcome"
REPORT_NAME = "execution-withdrawal-preoutcome-attempt2.json"
POINTER = BASE / "held-v8-attempt-2-withdrawal-pointer.json"
CODE_DIR_NAME = "code-90c5013e592a5808d73301e9451c896824a55974"
FAILED_CASE = "072-cotton-clohesline-ep0003"
EXPECTED_FAILURE = "ValueError: object is outside the dense panel"
COMPLETED_CASES = (
    "083-blanket-cloth-ep0003",
    "083-blanket-cloth-ep0006",
    "092-squirrel-ep0002",
    "092-squirrel-ep0003",
    "092-squirrel-ep0006",
    "170-spider-ep0004",
    "170-spider-ep0007",
)

EXPECTED = {
    "lock_artifact_sha256": "d8861839b8f7d0eecc4d3a17989276c917ff1ad85f75fd5ce3069dd126a913d0",
    "lock_file_sha256": "90cd30ae179377c4043496f0a32f958931d08018d4c02831b715b8c94fc4bfde",
    "source_artifact_sha256": "999f76ccd831def6afe765395c165bf4b95c8c0d8b3e2e860e79e482c71dfffa",
    "source_file_sha256": "b8011c9373244528cecd5d33cd73cac68f371dff907ef7d1bb801988f58b3765",
    "manifest_scale_artifact_sha256": "96f7edc666cda3cf84c6121623028c290b577ceec62cc104a41780b7bb6560ce",
    "manifest_scale_file_sha256": "3166d488258f1f62535c87813bbd895c9e4ba9855d43fa4393b8795f85c78973",
    "admission_artifact_sha256": "e659ceb9b4120c9a2e0c2bf33cbc8478bfc0157ed9b4f9415c3ebef194ea3f80",
    "admission_file_sha256": "ba45b56d1e127099d7ef1a910d199cc0f6c9dd698b7f785828163bc28904e2fb",
    "failure_log_sha256": "e296021c5b647d5e26cbf8cecd2e3fc46ebed97026a2564224a54f0fcd156b1c",
    "code_tree_sha256": "8b6bad43d33f4fd63257fc4c0a73965c07391b6a459623017fa022852829d906",
    "deployed_head": "90c5013e592a5808d73301e9451c896824a55974",
    "attempt1_pointer_file_sha256": "f7af6d1adf8541fd015cbe5336da97e013777c1bb711deaa01d9a84a49c81daa",
    "attempt1_report_file_sha256": "c04a6e7a95d958950ea7e7c05e7e2b98ee4516c01f03e9284f85ccccf0f6873b",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed_json_record(
    path: Path,
    *,
    root: Path,
    expected_file_sha256: str,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    state = os.lstat(path)
    _require(stat.S_ISREG(state.st_mode), f"not a regular file: {path}")
    _require(stat.S_IMODE(state.st_mode) == 0o400, f"not mode 0400: {path}")
    file_sha256 = _sha256_file(path)
    _require(file_sha256 == expected_file_sha256, f"file identity changed: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if expected_artifact_sha256 is not None:
        _require(
            data.get("artifact_sha256") == expected_artifact_sha256
            and _artifact_sha256(data) == expected_artifact_sha256,
            f"artifact identity changed: {path}",
        )
    return {
        "path": str(path.relative_to(root)),
        "mode_octal": "0400",
        "size_bytes": state.st_size,
        "file_sha256": file_sha256,
        **(
            {"artifact_sha256": expected_artifact_sha256}
            if expected_artifact_sha256 is not None
            else {}
        ),
    }


def _process_ancestry() -> set[int]:
    result: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            pid = int(fields[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            break
    return result


def _running_formal_processes() -> list[dict[str, Any]]:
    # Match only executable formal entry points.  Do not match a shell merely
    # because its here-document happens to contain the held-root path.
    entry_points = (
        "run_deform360_v8_calibration_shard.sh",
        "run_deform360_v8_calibration_case.sh",
        "run_deform360_v8_calibration_outcomes.py",
        "run_deform360_v8_confirmation_shard.sh",
        "run_deform360_v8_confirmation_case.sh",
        "acquire_deform360_v8_replacement_source.py",
    )
    own = _process_ancestry()
    records = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            argv = [
                part.decode(errors="replace")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        basenames = {Path(part).name for part in argv}
        if any(name in basenames for name in entry_points):
            records.append({"pid": int(entry.name), "argv": argv})
    return sorted(records, key=lambda row: row["pid"])


def _inventory(root: Path) -> dict[str, Any]:
    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for current, dir_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        dir_names[:] = sorted(
            name
            for name in dir_names
            if not (
                relative_current == Path(".") and name == CODE_DIR_NAME
            )
        )
        for name in dir_names:
            path = current_path / name
            state = os.lstat(path)
            _require(stat.S_ISDIR(state.st_mode), f"non-directory in walk: {path}")
            directories.append(
                {
                    "path": str(path.relative_to(root)),
                    "mode_octal": f"{stat.S_IMODE(state.st_mode):04o}",
                }
            )
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root)
            if relative == Path(REPORT_NAME):
                continue
            state = os.lstat(path)
            _require(not stat.S_ISLNK(state.st_mode), f"symlink present: {path}")
            _require(stat.S_ISREG(state.st_mode), f"special file present: {path}")
            files.append(
                {
                    "path": str(relative),
                    "mode_octal": f"{stat.S_IMODE(state.st_mode):04o}",
                    "size_bytes": state.st_size,
                    "sha256": _sha256_file(path),
                }
            )
    directories.sort(key=lambda row: row["path"])
    files.sort(key=lambda row: row["path"])
    payload = {"directories": directories, "regular_files": files}
    return {
        **payload,
        "directory_count": len(directories),
        "regular_file_count": len(files),
        "regular_file_bytes": sum(row["size_bytes"] for row in files),
        "inventory_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        "excluded_deployed_code_directory": CODE_DIR_NAME,
    }


def _count_execution_artifacts(root: Path) -> dict[str, Any]:
    cases = root / "calibration" / "cases"
    case_names = sorted(path.name for path in cases.iterdir() if path.is_dir())
    result = {
        "case_directories": case_names,
        "frame_zero_manifest_count": len(
            list(cases.glob("*/frame-zero/frame_zero_bundle.manifest.json"))
        ),
        "physical_prior_seal_count": len(
            list(cases.glob("*/physical/physical_prior_seal.json"))
        ),
        "prefix_authorization_count": len(
            list(cases.glob("*/prefix-authorization.json"))
        ),
        "online_prediction_seal_count": len(
            list(cases.glob("*/online/online_prediction_seal.json"))
        ),
        "frozen_field_manifest_count": len(
            list(cases.glob("*/frozen-field/preoutcome-frozen-field-manifest.json"))
        ),
    }
    _require(
        result
        == {
            "case_directories": [FAILED_CASE, *COMPLETED_CASES],
            "frame_zero_manifest_count": 8,
            "physical_prior_seal_count": 7,
            "prefix_authorization_count": 7,
            "online_prediction_seal_count": 7,
            "frozen_field_manifest_count": 7,
        },
        f"formal execution counts changed: {result}",
    )
    return result


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, 0o400)


def _make_tree_read_only(root: Path) -> None:
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        state = os.lstat(path)
        _require(not stat.S_ISLNK(state.st_mode), f"archive contains symlink: {path}")
        if stat.S_ISDIR(state.st_mode):
            os.chmod(path, 0o500)
        elif stat.S_ISREG(state.st_mode):
            os.chmod(path, 0o400)
        else:
            raise RuntimeError(f"archive contains special file: {path}")
    os.chmod(root, 0o500)
    _require(
        not root.stat().st_mode & 0o222
        and not any(path.stat().st_mode & 0o222 for path in root.rglob("*")),
        "archive remains writable",
    )


def main() -> int:
    _require(socket.gethostname() == "workstation2", "must run on gpuserver6000")
    _require(ACTIVE.is_dir() and not ACTIVE.is_symlink(), "active root is invalid")
    _require(not ARCHIVE.exists() and not POINTER.exists(), "archive/pointer exists")
    running = _running_formal_processes()
    _require(not running, f"formal held-v8 processes remain: {running}")
    _require((ACTIVE / CODE_DIR_NAME).is_dir(), "deployed code is absent")

    report_path = ACTIVE / REPORT_NAME
    _require(not report_path.exists(), "withdrawal report already exists")
    counts = _count_execution_artifacts(ACTIVE)
    forbidden_globs = (
        "calibration/**/*target*",
        "calibration/**/*query*",
        "calibration/**/*score*",
        "calibration/**/*decision*",
        "calibration/**/*barrier*",
        "confirmation*",
    )
    forbidden_found = sorted(
        str(path.relative_to(ACTIVE))
        for pattern in forbidden_globs
        for path in ACTIVE.glob(pattern)
    )
    _require(not forbidden_found, f"outcome/confirmation artifacts exist: {forbidden_found}")

    evidence = {
        "calibration_lock": _sealed_json_record(
            ACTIVE / "calibration-lock.json",
            root=ACTIVE,
            expected_file_sha256=EXPECTED["lock_file_sha256"],
            expected_artifact_sha256=EXPECTED["lock_artifact_sha256"],
        ),
        "replacement_source_manifest": _sealed_json_record(
            ACTIVE / "replacement-source/manifests/aligned-source.json",
            root=ACTIVE,
            expected_file_sha256=EXPECTED["source_file_sha256"],
            expected_artifact_sha256=EXPECTED["source_artifact_sha256"],
        ),
        "manifest_scale_diagnostic": _sealed_json_record(
            ACTIVE / "prewithdrawal-072-manifest-scale-diagnostic.json",
            root=ACTIVE,
            expected_file_sha256=EXPECTED["manifest_scale_file_sha256"],
            expected_artifact_sha256=EXPECTED["manifest_scale_artifact_sha256"],
        ),
        "admission_compatibility_diagnostic": _sealed_json_record(
            ACTIVE / "prewithdrawal-072-admission-compatibility-diagnostic.json",
            root=ACTIVE,
            expected_file_sha256=EXPECTED["admission_file_sha256"],
            expected_artifact_sha256=EXPECTED["admission_artifact_sha256"],
        ),
        "failure_log": {
            "path": f"calibration/logs/{FAILED_CASE}.physical.failed.log",
            "sha256": _sha256_file(
                ACTIVE / "calibration/logs" / f"{FAILED_CASE}.physical.failed.log"
            ),
        },
        "attempt1_pointer": {
            "path": str(BASE / "held-v8-attempt-1-withdrawal-pointer.json"),
            "file_sha256": _sha256_file(
                BASE / "held-v8-attempt-1-withdrawal-pointer.json"
            ),
        },
        "attempt1_report": {
            "path": str(
                BASE
                / "held-v8-attempt-1-withdrawn-preoutcome"
                / "execution-withdrawal-preoutcome.json"
            ),
            "file_sha256": _sha256_file(
                BASE
                / "held-v8-attempt-1-withdrawn-preoutcome"
                / "execution-withdrawal-preoutcome.json"
            ),
        },
        "deployed_code": {
            "path": CODE_DIR_NAME,
            "git_head": EXPECTED["deployed_head"],
            "filesystem_tree_sha256": EXPECTED["code_tree_sha256"],
        },
    }
    _require(
        evidence["failure_log"]["sha256"] == EXPECTED["failure_log_sha256"],
        "failure log identity changed",
    )
    _require(
        evidence["attempt1_pointer"]["file_sha256"]
        == EXPECTED["attempt1_pointer_file_sha256"],
        "attempt-1 pointer identity changed",
    )
    _require(
        evidence["attempt1_report"]["file_sha256"]
        == EXPECTED["attempt1_report_file_sha256"],
        "attempt-1 report identity changed",
    )
    failure_text = (
        ACTIVE / "calibration/logs" / f"{FAILED_CASE}.physical.failed.log"
    ).read_text(encoding="utf-8")
    _require(EXPECTED_FAILURE in failure_text, "failure signature changed")

    first_inventory = _inventory(ACTIVE)
    second_inventory = _inventory(ACTIVE)
    _require(first_inventory == second_inventory, "evidence inventory changed between passes")

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldV8Attempt2ExecutionWithdrawalPreoutcome",
        "protocol_id": "deform360-held-online-belief-v8",
        "status": "withdrawn-preoutcome-after-partial-prediction",
        "date": "2026-07-22",
        "formal_root_before_withdrawal": str(ACTIVE),
        "immutable_archive_path": str(ARCHIVE),
        "evidence_bindings": evidence,
        "execution": {
            **counts,
            "shard_0": {
                "started_cases": [FAILED_CASE],
                "terminal_state": "failed-during-physical-build",
            },
            "shard_1": {
                "completed_cases": list(COMPLETED_CASES),
                "terminal_state": "SHARD_COMPLETE",
            },
            "failure": {
                "case_name": FAILED_CASE,
                "phase": "physical-build",
                "signature": EXPECTED_FAILURE,
                "rollout_reached": False,
                "physical_seal_created": False,
                "cause": (
                    "The frozen inherited automatic-twin runtime admitted only its "
                    "legacy dense-panel cohort and rejected the preregistered external "
                    "cotton calibration episode before simulation."
                ),
            },
            "partial_predictions_disposition": (
                "Seven online predictions existed. Their numerical arrays were not "
                "opened or selected during withdrawal/remediation and none may be "
                "reused by a successor attempt."
            ),
        },
        "stable_evidence_inventory": {
            **first_inventory,
            "stable_two_pass_match": True,
            "report_excluded_because_created_after_inventory": REPORT_NAME,
        },
        "information_boundary": {
            "causal_rgb_prefixes_consumed_by_completed_predictions": True,
            "known_robot_kinematics_may_have_been_consumed_by_physical_construction": True,
            "prediction_arrays_inspected_during_remediation": False,
            "official_future_target_coordinates_read": False,
            "official_future_target_masks_or_visibility_read": False,
            "official_future_target_rgb_or_geometry_read": False,
            "target_reconstructed": False,
            "query_output_created_or_read": False,
            "score_computed_or_read": False,
            "first_cohort_barrier_created_or_crossed": False,
            "second_cohort_barrier_created_or_crossed": False,
            "calibration_gate_created_or_read": False,
            "confirmation_lock_created": False,
            "confirmation_payload_accessed": False,
            "method_selection_informed_by_attempt2_outcome": False,
        },
        "successor_disposition": {
            "reuse_attempt2_predictions_fields_or_partial_case_artifacts": False,
            "reuse_attempt2_source_permit": False,
            "fresh_full_fifteen_case_rerun_required": True,
            "allowed_changes": [
                "an exact-case v8 admission adapter for 072-cotton-clohesline-ep0003",
                "a bounded digest-based representation of the unchanged exhaustive frame-zero audit",
            ],
            "unchanged": [
                "calibration and confirmation cohorts",
                "models and checkpoints",
                "primary method and physical numerics",
                "query field and hyperparameters",
                "gates and metrics",
                "outcome operators",
                "frozen upstream runtimes",
            ],
        },
    }
    report["artifact_sha256"] = _artifact_sha256(report)
    _exclusive_json(report_path, report)
    _require(
        _artifact_sha256(json.loads(report_path.read_text()))
        == report["artifact_sha256"],
        "written report artifact hash changed",
    )
    report_file_sha256 = _sha256_file(report_path)

    os.rename(ACTIVE, ARCHIVE)
    archived_report = ARCHIVE / REPORT_NAME
    _require(archived_report.is_file(), "report did not move with archive")
    _make_tree_read_only(ARCHIVE)

    pointer: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldV8Attempt2WithdrawalPointer",
        "protocol_id": "deform360-held-online-belief-v8",
        "status": "withdrawn-preoutcome-after-partial-prediction",
        "date": "2026-07-22",
        "archive_path": str(ARCHIVE),
        "archive_fully_nonwritable": True,
        "active_held_v8_root_absent_after_archive": not ACTIVE.exists(),
        "withdrawal_report_path": str(archived_report),
        "withdrawal_report_file_sha256": report_file_sha256,
        "withdrawal_report_artifact_sha256": report["artifact_sha256"],
        "calibration_outcome_observed": False,
        "confirmation_accessed": False,
    }
    pointer["artifact_sha256"] = _artifact_sha256(pointer)
    _exclusive_json(POINTER, pointer)
    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "report_file_sha256": report_file_sha256,
                "report_artifact_sha256": report["artifact_sha256"],
                "pointer_file_sha256": _sha256_file(POINTER),
                "pointer_artifact_sha256": pointer["artifact_sha256"],
                "inventory_file_count": first_inventory["regular_file_count"],
                "inventory_bytes": first_inventory["regular_file_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
