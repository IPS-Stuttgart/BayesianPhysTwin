#!/usr/bin/env python3
"""Seal the consumed held-v8.2 child-runtime failure without opening outcomes."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(SOURCE_ROOT / "src"))
from bayesian_phystwin import (  # noqa: E402
    deform360_held_v82_technical_failure as integrity,
)


BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
ACTIVE = BASE / "held-v82"
ARCHIVE = BASE / "held-v82-attempt-1-technical-failure"
POINTER = BASE / "held-v82-attempt-1-technical-failure-pointer.json"
COMPLETION = BASE / "held-v82-attempt-1-technical-failure-completion.json"
OPERATOR_SOURCE = Path(__file__).resolve()
EXPECTED_HOST = "workstation2"
EXPECTED_DEPLOYED_HEAD = "7a3d2beaf45e5068f114ea9dd20ebbce16e5b4b8"
EXPECTED_DEPLOYED_CODE_NAME = f"code-{EXPECTED_DEPLOYED_HEAD}"
EXPECTED_LOCK_FILE_SHA256 = (
    "f0af7f04c7483c341e7b6af9b577cb3fbaae02cd550e6b74d2942dfefb73c04e"
)
EXPECTED_LOCK_ARTIFACT_SHA256 = (
    "801c6a3b69d1f86c4cf138a567c45bd9537701c0759f77588445522cdb638c25"
)
FAILED_CASE = "072-cotton-clohesline-ep0003"
EXPECTED_STDOUT_SHA256 = (
    "d23146cb8196224d394b6193b6d829bbb8ce40d8019c8b1886d3d1c37915885e"
)
EXPECTED_STDERR_SHA256 = (
    "62fc314b264b0732edbcd375e95566e2d4b43bcc3c1b9bcd087d00163ddcaa95"
)
EXPECTED_STDOUT_SIZE = 2171
EXPECTED_STDERR_SIZE = 18533
ERROR_MARKERS = (
    b"AttributeError",
    b"'NoneType' object has no attribute 'CameraModelType'",
    b"gsplat/cuda/_wrapper.py",
)
EXPECTED_CASES = (
    "072-cotton-clohesline-ep0003",
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
FORBIDDEN_NAMES = frozenset(
    {
        "official-target-manifest.json",
        "official-target.npz",
        "official-frame-zero-query-manifest.json",
        "official-frame-zero-query.npz",
        "queried-prediction-seal.json",
        "queried-prediction.npz",
        "calibration-score-evidence.json",
        "calibration-gate-decision.json",
        "confirmation-lock.json",
        "confirmation-score-evidence.json",
        "confirmation-gate-decision.json",
        "role-execution-completion.json",
    }
)
FORMAL_PROCESS_MARKERS = (
    "held-v82",
    "run_deform360_v8_calibration",
    "run_deform360_v8_confirmation",
    "deform360_held_v8_outcome_driver",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    observed = integrity.stable_file(path, collect=True)
    payload = observed.pop("payload")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} is not UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{role} is not a JSON object")
    return value


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _pretty_bytes(value)
    if os.path.lexists(path):
        observed = integrity.stable_file(path, collect=True, required_mode=0o400)
        existing = observed.pop("payload")
        _require(existing == payload, f"existing sealed JSON changed: {path}")
        return observed
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    return integrity.stable_file(path, required_mode=0o400)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _process_ancestry() -> set[int]:
    result: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
            pid = int(fields[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            break
    return result


def _running_formal_processes() -> list[dict[str, Any]]:
    own = _process_ancestry()
    records: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return records
    for entry in proc.iterdir():
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
        joined = " ".join(argv)
        if any(marker in joined for marker in FORMAL_PROCESS_MARKERS):
            records.append({"pid": int(entry.name), "argv": argv})
    return sorted(records, key=lambda row: int(row["pid"]))


def _run_git(code: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", os.fspath(code), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/home/florianpfaff",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
    )
    _require(
        completed.returncode == 0,
        f"git {' '.join(arguments)} failed: {completed.stderr.strip()}",
    )
    return completed.stdout.strip()


def _deployed_code(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.startswith("code-")
    ]
    _require(
        len(candidates) == 1 and candidates[0].name == EXPECTED_DEPLOYED_CODE_NAME,
        "deployed v8.2 code directory changed",
    )
    code = candidates[0]
    head = _run_git(code, "rev-parse", "HEAD").lower()
    tree = _run_git(code, "rev-parse", "HEAD^{tree}").lower()
    status = _run_git(code, "status", "--porcelain=v1", "--untracked-files=all")
    _require(
        head == EXPECTED_DEPLOYED_HEAD and not status,
        "deployed v8.2 source identity changed",
    )
    bindings = lock.get("immutable_bindings")
    _require(isinstance(bindings, Mapping), "v8.2 lock bindings are absent")
    _require(
        bindings.get("method_head_text_sha256")
        == hashlib.sha256(head.encode("ascii")).hexdigest(),
        "deployed v8.2 HEAD differs from its lock",
    )
    return {
        "path": code.name,
        "git_head": head,
        "git_tree": tree,
        "head_text_sha256": hashlib.sha256(head.encode("ascii")).hexdigest(),
        "worktree_clean": True,
    }


def _lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "calibration-lock.json"
    observed = integrity.stable_file(path, collect=True, required_mode=0o400)
    payload = observed.pop("payload")
    _require(
        observed["sha256"] == EXPECTED_LOCK_FILE_SHA256,
        "v8.2 calibration lock file changed",
    )
    value = json.loads(payload.decode("utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("artifact_sha256") == EXPECTED_LOCK_ARTIFACT_SHA256
        and integrity.artifact_sha256(value) == EXPECTED_LOCK_ARTIFACT_SHA256
        and value.get("protocol_id") == integrity.PROTOCOL_ID
        and value.get("execution_attempt") == integrity.EXECUTION_ATTEMPT
        and value.get("stage") == "calibration"
        and value.get("held_root") == os.fspath(ACTIVE)
        and tuple(value.get("calibration_case_whitelist", ())) == EXPECTED_CASES,
        "v8.2 calibration lock identity changed",
    )
    return value, observed


def _validate_boundary(root: Path) -> dict[str, Any]:
    cases = root / "calibration/cases"
    observed_cases = sorted(path.name for path in cases.iterdir() if path.is_dir())
    _require(observed_cases == sorted(EXPECTED_CASES), "v8.2 case cohort changed")
    stage_paths = (
        "frame-zero/frame_zero_bundle.manifest.json",
        "physical/physical_prior_seal.json",
        "prefix-authorization.json",
        "online/online_prediction_seal.json",
        "frozen-field/preoutcome-frozen-field-manifest.json",
    )
    for relative in stage_paths:
        _require(
            all(
                (cases / case / relative).is_file()
                and stat.S_IMODE(os.lstat(cases / case / relative).st_mode) == 0o400
                for case in EXPECTED_CASES
            ),
            f"pre-outcome source stage is incomplete: {relative}",
        )
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name in FORBIDDEN_NAMES
        or path.name.startswith("confirmation-")
        or path.name.endswith("-score-evidence.json")
    ]
    _require(not forbidden, f"protected v8.2 outcome artifact exists: {forbidden}")
    private = root / "calibration/private-targets"
    query_inputs = root / "calibration/query-inputs"
    query_outputs = root / "calibration/query-outputs"
    for case in EXPECTED_CASES:
        private_names = {entry.name for entry in (private / case).iterdir()}
        if case == FAILED_CASE:
            _require(
                private_names
                == {
                    "fresh-official-reconstruction",
                    "isolated-official-reconstruction.stdout.log",
                    "isolated-official-reconstruction.stderr.log",
                },
                "failed-case source-side evidence changed",
            )
        else:
            _require(not private_names, f"unexpected private artifact exists: {case}")
        _require(
            not any((query_inputs / case).iterdir())
            and not any((query_outputs / case).iterdir()),
            f"query artifact exists before the v8.2 technical failure: {case}",
        )
    reconstruction = private / FAILED_CASE / "fresh-official-reconstruction"
    _require(
        reconstruction.is_dir()
        and {entry.name for entry in reconstruction.iterdir()} == {"staged-aligned"},
        "partial source reconstruction boundary changed",
    )
    stdout = integrity.stable_file(
        private / FAILED_CASE / "isolated-official-reconstruction.stdout.log",
        required_mode=0o400,
    )
    stderr = integrity.stable_file(
        private / FAILED_CASE / "isolated-official-reconstruction.stderr.log",
        collect=True,
        required_mode=0o400,
    )
    stderr_payload = stderr.pop("payload")
    _require(
        stdout["sha256"] == EXPECTED_STDOUT_SHA256
        and stdout["size_bytes"] == EXPECTED_STDOUT_SIZE
        and stderr["sha256"] == EXPECTED_STDERR_SHA256
        and stderr["size_bytes"] == EXPECTED_STDERR_SIZE
        and all(stderr_payload.count(marker) >= 1 for marker in ERROR_MARKERS),
        "v8.2 child-runtime failure evidence changed",
    )
    return {
        "calibration_case_count": len(EXPECTED_CASES),
        "complete_preoutcome_source_case_count": len(EXPECTED_CASES),
        "partial_source_reconstruction_count": 1,
        "failed_case": FAILED_CASE,
        "failure_stdout": stdout,
        "failure_stderr": stderr,
        "official_target_artifact_count": 0,
        "official_query_artifact_count": 0,
        "queried_prediction_artifact_count": 0,
        "score_artifact_count": 0,
        "gate_decision_count": 0,
        "confirmation_artifact_count": 0,
    }


def _operator_source() -> dict[str, Any]:
    _require(
        OPERATOR_SOURCE.name == "seal_deform360_v82_technical_failure.py"
        and OPERATOR_SOURCE.is_absolute()
        and OPERATOR_SOURCE.resolve() == OPERATOR_SOURCE,
        "technical-failure operator source path changed",
    )
    return integrity.stable_file(OPERATOR_SOURCE)


def _seal_noncode_tree(root: Path, *, code_name: str) -> None:
    for current, directories, files in os.walk(root, topdown=False):
        parent = Path(current)
        relative = parent.relative_to(root)
        if relative.parts and relative.parts[0] == code_name:
            continue
        for name in files:
            os.chmod(parent / name, 0o400, follow_symlinks=False)
        for name in directories:
            child = parent / name
            if relative == Path(".") and name == code_name:
                continue
            os.chmod(child, 0o500, follow_symlinks=False)
    os.chmod(root, 0o500, follow_symlinks=False)


def _file_binding(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    observed = integrity.stable_file(path, required_mode=0o400)
    return {
        **observed,
        "artifact_sha256": value["artifact_sha256"],
    }


def _build_completion(
    report: Mapping[str, Any],
    report_binding: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    return integrity.signed(
        {
            "schema_version": 1,
            "artifact_kind": integrity.COMPLETION_KIND,
            "protocol_id": integrity.PROTOCOL_ID,
            "execution_attempt": integrity.EXECUTION_ATTEMPT,
            "status": integrity.STATUS,
            "result_status": integrity.RESULT_STATUS,
            "immutable_archive_path": os.fspath(ARCHIVE),
            "report": dict(report_binding),
            "archive_inventory": dict(inventory),
            "deployed_code": dict(report["deployed_code"]),
            "archive_root_mode_octal": "0500",
            "integrity_complete": True,
        }
    )


def _build_pointer(
    report_binding: Mapping[str, Any],
    completion_binding: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    return integrity.signed(
        {
            "schema_version": 1,
            "artifact_kind": integrity.POINTER_KIND,
            "protocol_id": integrity.PROTOCOL_ID,
            "execution_attempt": integrity.EXECUTION_ATTEMPT,
            "status": integrity.STATUS,
            "result_status": integrity.RESULT_STATUS,
            "immutable_archive_path": os.fspath(ARCHIVE),
            "report": dict(report_binding),
            "completion": dict(completion_binding),
            "archive_inventory_sha256": inventory["inventory_sha256"],
            "successor_root_reuse_permitted": False,
        }
    )


def seal() -> dict[str, Any]:
    _require(socket.gethostname() == EXPECTED_HOST, "formal seal must run on workstation2")
    _require(
        not (os.path.lexists(ACTIVE) and os.path.lexists(ARCHIVE)),
        "active and archived v8.2 roots both exist",
    )
    _require(not _running_formal_processes(), "a formal Deform360 process is running")
    operator = _operator_source()

    if os.path.lexists(ACTIVE):
        lock, lock_record = _lock(ACTIVE)
        code = _deployed_code(ACTIVE, lock)
        boundary = _validate_boundary(ACTIVE)
        first_inventory = integrity.archive_inventory(
            ACTIVE,
            excluded_code_directory=code["path"],
            exclude_report=True,
        )
        second_inventory = integrity.archive_inventory(
            ACTIVE,
            excluded_code_directory=code["path"],
            exclude_report=True,
        )
        _require(first_inventory == second_inventory, "v8.2 root changed during inventory")
        report = integrity.signed(
            {
                "schema_version": 1,
                "artifact_kind": integrity.REPORT_KIND,
                "protocol_id": integrity.PROTOCOL_ID,
                "execution_attempt": integrity.EXECUTION_ATTEMPT,
                "status": integrity.STATUS,
                "result_status": integrity.RESULT_STATUS,
                "date": "2026-07-25",
                "formal_root_before_failure_seal": os.fspath(ACTIVE),
                "immutable_archive_path": os.fspath(ARCHIVE),
                "executed_operator_source": operator,
                "deployed_code": code,
                "calibration_lock": {
                    **lock_record,
                    "artifact_sha256": lock["artifact_sha256"],
                },
                "terminal_failure": {
                    "failure_phase": (
                        "first isolated official reconstruction before target or query "
                        "materialization"
                    ),
                    "failed_case": FAILED_CASE,
                    "exception_type": "AttributeError",
                    "exception_message_class": (
                        "gsplat compiled backend absent in the isolated child process"
                    ),
                    "root_cause": (
                        "child runtime did not preload the frozen AOT gsplat extension"
                    ),
                },
                "execution_boundary": boundary,
                "stable_pre_report_noncode_inventory": first_inventory,
                "information_boundary": {
                    "first_complete_cohort_barrier_crossed": True,
                    "official_target_artifact_created_or_read": False,
                    "official_query_artifact_created_or_read": False,
                    "queried_prediction_created_or_read": False,
                    "score_created_or_read": False,
                    "gate_decision_created_or_read": False,
                    "confirmation_created_or_read": False,
                    "failure_evidence_is_source_side_runtime_only": True,
                    "forensic_operator_deserialized_protected_outcome_payload": False,
                },
                "claim_boundary": (
                    "This is a technical execution failure with no calibration result "
                    "and no evidence for or against the frozen method."
                ),
                "successor_policy": {
                    "reuse_v82_root": False,
                    "reuse_v82_predictions_or_partial_reconstruction": False,
                    "fresh_revision_and_fresh_root_required": True,
                },
            }
        )
        _exclusive_json(ACTIVE / integrity.REPORT_NAME, report)
        os.rename(ACTIVE, ARCHIVE)
        _fsync_directory(BASE)

    _require(ARCHIVE.is_dir(), "sealed v8.2 archive is absent")
    report_path = ARCHIVE / integrity.REPORT_NAME
    report, _ = integrity.load_signed(report_path, role="v8.2 failure report")
    _require(
        report.get("immutable_archive_path") == os.fspath(ARCHIVE),
        "archived v8.2 report identity changed",
    )
    code_name = str(report["deployed_code"]["path"])
    _seal_noncode_tree(ARCHIVE, code_name=code_name)
    final_inventory = integrity.archive_inventory(
        ARCHIVE,
        excluded_code_directory=code_name,
    )
    report_binding = _file_binding(report_path, report)
    completion = _build_completion(report, report_binding, final_inventory)
    _exclusive_json(COMPLETION, completion)
    completion_binding = _file_binding(COMPLETION, completion)
    pointer = _build_pointer(report_binding, completion_binding, final_inventory)
    _exclusive_json(POINTER, pointer)
    return integrity.validate_v82_technical_failure_lineage(
        archive_path=ARCHIVE,
        report_path=report_path,
        pointer_path=POINTER,
        completion_path=COMPLETION,
        verify_content_inventory=True,
    )


def main() -> int:
    result = seal()
    print(
        json.dumps(
            {
                "operation": "sealed_deform360_v82_technical_failure",
                "archive": result["v82_technical_failure_archive_integrity"]["path"],
                "inventory_sha256": result[
                    "v82_technical_failure_archive_integrity"
                ]["inventory_sha256"],
                "result_status": result["v82_calibration_result"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
