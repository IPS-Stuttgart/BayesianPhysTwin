#!/usr/bin/env python3
"""Complete the immutable integrity audit of the held-v8 attempt-2 archive.

The original withdrawal report was sealed before the active root was renamed.
This separate, immutable completion validates the resulting archive with
no-follow stable reads, verifies every recorded evidence hash, recomputes the
deployed Git tree from the archived working files, and records exact absence of
all outcome-phase roots.  It never opens prediction array contents.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
from typing import Any, Mapping


BASE = Path("/mnt/corsair/florianpfaff/bpt-online-belief-v1")
ACTIVE = BASE / "held-v8"
ARCHIVE = BASE / "held-v8-attempt-2-withdrawn-preoutcome"
REPORT = ARCHIVE / "execution-withdrawal-preoutcome-attempt2.json"
POINTER = BASE / "held-v8-attempt-2-withdrawal-pointer.json"
OUTPUT = BASE / "held-v8-attempt-2-withdrawal-integrity-completion.json"
CODE_NAME = "code-90c5013e592a5808d73301e9451c896824a55974"
EXPECTED_HEAD = "90c5013e592a5808d73301e9451c896824a55974"
EXPECTED_GIT_TREE_MANIFEST_SHA256 = (
    "8b6bad43d33f4fd63257fc4c0a73965c07391b6a459623017fa022852829d906"
)
EXPECTED_REPORT_FILE_SHA256 = (
    "5830f9bfe8d29d5a09f64afbcaeabadc3acb7c8fdf820c1aeb68a6601055a895"
)
EXPECTED_REPORT_ARTIFACT_SHA256 = (
    "457c6a64c0208b91ee5eb0f8038d22ae7eda743e29fb60a4bcb4ef1a2861b147"
)
EXPECTED_POINTER_FILE_SHA256 = (
    "007d3fbde0dc93dc350661aafdd5d08d1398aa8d1f164e17bf295521fc40463a"
)
EXPECTED_POINTER_ARTIFACT_SHA256 = (
    "9063011657b955902d1cf7d85a4253eee65caa430a41edae2709a18032baf99c"
)
ATTEMPT1 = {
    "pointer": (
        BASE / "held-v8-attempt-1-withdrawal-pointer.json",
        "f7af6d1adf8541fd015cbe5336da97e013777c1bb711deaa01d9a84a49c81daa",
        "7a0e966acc23362afead3b1b78f433b473ae846542240df050a408653572b8e3",
    ),
    "report": (
        BASE
        / "held-v8-attempt-1-withdrawn-preoutcome"
        / "execution-withdrawal-preoutcome.json",
        "c04a6e7a95d958950ea7e7c05e7e2b98ee4516c01f03e9284f85ccccf0f6873b",
        "53064dcd770f48acd4d828112853bcd34da81336df42648fd577f264d3ebaac9",
    ),
}


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


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _stable_identity(state: os.stat_result) -> tuple[int, ...]:
    return (
        state.st_dev,
        state.st_ino,
        stat.S_IFMT(state.st_mode),
        stat.S_IMODE(state.st_mode),
        state.st_size,
        state.st_mtime_ns,
        state.st_ctime_ns,
    )


def _stable_file(
    path: Path,
    *,
    required_mode: int = 0o400,
    collect: bool = False,
    git_blob: bool = False,
) -> dict[str, Any]:
    before = os.lstat(path)
    _require(
        stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode),
        f"not a regular non-symlink file: {path}",
    )
    _require(stat.S_IMODE(before.st_mode) == required_mode, f"mode changed: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    sha256 = hashlib.sha256()
    blob = hashlib.sha1() if git_blob else None
    if blob is not None:
        blob.update(f"blob {before.st_size}\0".encode("ascii"))
    payload = bytearray() if collect else None
    try:
        opened = os.fstat(descriptor)
        _require(_stable_identity(opened) == _stable_identity(before), f"open race: {path}")
        while block := os.read(descriptor, 8 * 1024 * 1024):
            sha256.update(block)
            if blob is not None:
                blob.update(block)
            if payload is not None:
                payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(path)
    _require(
        _stable_identity(before)
        == _stable_identity(after)
        == _stable_identity(current),
        f"file changed while hashing: {path}",
    )
    return {
        "size_bytes": before.st_size,
        "sha256": sha256.hexdigest(),
        **({"git_blob_object_id": blob.hexdigest()} if blob is not None else {}),
        **({"payload": bytes(payload)} if payload is not None else {}),
    }


def _sealed_json(
    path: Path, *, file_sha256: str, artifact_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = _stable_file(path, collect=True)
    _require(observed["sha256"] == file_sha256, f"sealed file changed: {path}")
    try:
        value = json.loads(observed.pop("payload").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"sealed JSON is invalid: {path}") from error
    _require(
        value.get("artifact_sha256") == artifact_sha256
        and _artifact_sha256(value) == artifact_sha256,
        f"sealed artifact changed: {path}",
    )
    return value, {
        "path": str(path),
        "mode_octal": "0400",
        "size_bytes": observed["size_bytes"],
        "file_sha256": observed["sha256"],
        "artifact_sha256": artifact_sha256,
    }


def _parse_git_tree(raw: bytes) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        header, separator, path_bytes = encoded.partition(b"\t")
        _require(bool(separator) and bool(path_bytes), "malformed Git tree record")
        fields = header.split(b" ")
        _require(len(fields) == 3, "malformed Git tree header")
        mode, kind, object_id = (field.decode("ascii") for field in fields)
        path = path_bytes.decode("utf-8")
        _require(
            mode in {"100644", "100755"}
            and kind == "blob"
            and len(object_id) == 40
            and all(character in "0123456789abcdef" for character in object_id),
            f"unsupported Git record: {path}",
        )
        _require(
            path and not path.startswith("/") and ".." not in Path(path).parts,
            "unsafe Git path",
        )
        records.append(
            {"mode": mode, "type": kind, "object_id": object_id, "path": path}
        )
    _require(
        records and [row["path"] for row in records] == sorted(row["path"] for row in records),
        "Git records are absent or unsorted",
    )
    return records


def _git(code: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(code), *arguments],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "HOME": "/home/florianpfaff",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
    )
    _require(
        completed.returncode == 0,
        f"Git {' '.join(arguments)} failed: {completed.stderr.decode(errors='replace')}",
    )
    return completed.stdout


def _validate_code() -> dict[str, Any]:
    code = ARCHIVE / CODE_NAME
    state = os.lstat(code)
    _require(
        stat.S_ISDIR(state.st_mode)
        and not stat.S_ISLNK(state.st_mode)
        and stat.S_IMODE(state.st_mode) == 0o500,
        "archived deployment root is invalid",
    )
    head = _git(code, "rev-parse", "HEAD").decode().strip().lower()
    _require(head == EXPECTED_HEAD, "archived deployment HEAD changed")
    records = _parse_git_tree(_git(code, "ls-tree", "-r", "-z", "HEAD"))
    tree_sha256 = hashlib.sha256(_canonical_bytes(records)).hexdigest()
    _require(
        tree_sha256 == EXPECTED_GIT_TREE_MANIFEST_SHA256,
        "archived deployment Git tree manifest changed",
    )
    tracked = {row["path"] for row in records}
    working = {
        str(path.relative_to(code))
        for path in code.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(code).parts
    }
    _require(working == tracked, "archived deployment has missing or untracked files")
    for row in records:
        observed = _stable_file(code / row["path"], git_blob=True)
        _require(
            observed["git_blob_object_id"] == row["object_id"],
            f"archived tracked content changed: {row['path']}",
        )
    return {
        "path": str(code),
        "head": head,
        "tracked_file_count": len(records),
        "git_tree_manifest_sha256": tree_sha256,
        "every_working_file_matches_bound_git_blob": True,
        "no_untracked_working_files": True,
    }


def _validate_inventory(report: Mapping[str, Any]) -> dict[str, Any]:
    inventory = report["stable_evidence_inventory"]
    rows = inventory["regular_files"]
    directories = inventory["directories"]
    payload = {"directories": directories, "regular_files": rows}
    _require(
        hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        == inventory["inventory_sha256"],
        "withdrawal inventory digest changed",
    )
    expected_paths: set[str] = set()
    for row in rows:
        relative = Path(row["path"])
        _require(
            relative.parts
            and not relative.is_absolute()
            and ".." not in relative.parts
            and relative.parts[0] != CODE_NAME
            and relative != Path(REPORT.name),
            "unsafe inventory path",
        )
        observed = _stable_file(ARCHIVE / relative)
        _require(
            observed["size_bytes"] == row["size_bytes"]
            and observed["sha256"] == row["sha256"],
            f"inventoried file changed: {relative}",
        )
        expected_paths.add(str(relative))
    observed_paths = {
        str(path.relative_to(ARCHIVE))
        for path in ARCHIVE.rglob("*")
        if path.is_file()
        and path != REPORT
        and path.relative_to(ARCHIVE).parts[0] != CODE_NAME
    }
    _require(observed_paths == expected_paths, "non-code archive inventory is incomplete")
    expected_directories = {row["path"] for row in directories}
    observed_directories = {
        str(path.relative_to(ARCHIVE))
        for path in ARCHIVE.rglob("*")
        if path.is_dir() and path.relative_to(ARCHIVE).parts[0] != CODE_NAME
    }
    _require(
        observed_directories == expected_directories,
        "non-code archive directory inventory is incomplete",
    )
    return {
        "inventory_sha256": inventory["inventory_sha256"],
        "regular_file_count": len(rows),
        "regular_file_bytes": sum(row["size_bytes"] for row in rows),
        "directory_count": len(directories),
        "every_record_revalidated_with_stable_nofollow_read": True,
        "inventory_path_sets_exact": True,
    }


def _validate_archive_structure() -> dict[str, Any]:
    root = os.lstat(ARCHIVE)
    _require(
        stat.S_ISDIR(root.st_mode)
        and not stat.S_ISLNK(root.st_mode)
        and stat.S_IMODE(root.st_mode) == 0o500,
        "archive root is invalid",
    )
    directory_count = 1
    file_count = 0
    for current, directories, files in os.walk(ARCHIVE, followlinks=False):
        for name in directories:
            state = os.lstat(Path(current) / name)
            _require(
                stat.S_ISDIR(state.st_mode)
                and not stat.S_ISLNK(state.st_mode)
                and stat.S_IMODE(state.st_mode) == 0o500,
                f"archive directory mode/type changed: {Path(current) / name}",
            )
            directory_count += 1
        for name in files:
            state = os.lstat(Path(current) / name)
            _require(
                stat.S_ISREG(state.st_mode)
                and not stat.S_ISLNK(state.st_mode)
                and stat.S_IMODE(state.st_mode) == 0o400,
                f"archive file mode/type changed: {Path(current) / name}",
            )
            file_count += 1
    _require(not os.path.lexists(ACTIVE), "active held-v8 root reappeared")
    role = ARCHIVE / "calibration"
    forbidden = (
        role / ".v8-outcome-phase.claim",
        role / "private-targets",
        role / "query-inputs",
        role / "query-outputs",
        role / "calibration-score-evidence.json",
        role / "calibration-gate-decision.json",
        ARCHIVE / "confirmation-lock.json",
        ARCHIVE / "confirmation",
    )
    _require(
        not any(os.path.lexists(path) for path in forbidden),
        "outcome or confirmation root exists in archive",
    )
    return {
        "root_mode_octal": "0500",
        "directory_count_including_root": directory_count,
        "regular_file_count": file_count,
        "all_directories_mode_0500": True,
        "all_files_mode_0400": True,
        "symlink_count": 0,
        "special_file_count": 0,
        "active_root_absent": True,
        "outcome_claim_absent": True,
        "private_target_root_absent": True,
        "query_input_root_absent": True,
        "query_output_root_absent": True,
        "score_and_gate_artifacts_absent": True,
        "confirmation_lock_and_root_absent": True,
    }


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


def main() -> int:
    _require(socket.gethostname() == "workstation2", "must run on gpuserver6000")
    _require(ARCHIVE.is_dir() and not ARCHIVE.is_symlink(), "archive is absent")
    _require(not os.path.lexists(OUTPUT), "integrity completion already exists")
    report, report_record = _sealed_json(
        REPORT,
        file_sha256=EXPECTED_REPORT_FILE_SHA256,
        artifact_sha256=EXPECTED_REPORT_ARTIFACT_SHA256,
    )
    pointer, pointer_record = _sealed_json(
        POINTER,
        file_sha256=EXPECTED_POINTER_FILE_SHA256,
        artifact_sha256=EXPECTED_POINTER_ARTIFACT_SHA256,
    )
    _require(
        pointer["withdrawal_report_file_sha256"] == EXPECTED_REPORT_FILE_SHA256
        and pointer["withdrawal_report_artifact_sha256"]
        == EXPECTED_REPORT_ARTIFACT_SHA256,
        "attempt-2 pointer does not bind its report",
    )
    parent_records = {}
    for name, (path, file_sha256, artifact_sha256) in ATTEMPT1.items():
        _value, parent_records[name] = _sealed_json(
            path, file_sha256=file_sha256, artifact_sha256=artifact_sha256
        )

    withdrawal_operator = Path("/tmp/seal_deform360_v8_attempt2_withdrawal.py")
    completion_operator = Path(__file__).resolve()
    operator_records = {
        "attempt2_withdrawal_operator": {
            "path_at_execution": str(withdrawal_operator),
            **_stable_file(withdrawal_operator, required_mode=0o500),
        },
        "attempt2_integrity_completion_operator": {
            "path_at_execution": str(completion_operator),
            **_stable_file(completion_operator, required_mode=0o500),
        },
    }

    structure = _validate_archive_structure()
    inventory = _validate_inventory(report)
    code = _validate_code()
    completion: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldV8Attempt2WithdrawalIntegrityCompletion",
        "protocol_id": "deform360-held-online-belief-v8",
        "status": "withdrawal-integrity-complete",
        "date": "2026-07-22",
        "attempt2_withdrawal_report": report_record,
        "attempt2_withdrawal_pointer": pointer_record,
        "attempt1_lineage_revalidated": parent_records,
        "operator_source_bindings": operator_records,
        "archive_structure": structure,
        "stable_evidence_inventory_revalidation": inventory,
        "deployed_code_revalidation": code,
        "completion_scope": {
            "prediction_array_payloads_opened": False,
            "future_target_payloads_opened": False,
            "query_or_score_payloads_opened": False,
            "metadata_and_source_files_only": True,
            "addresses_report_reader_limitations": (
                "This completion uses stable O_NOFOLLOW reads and recomputes the "
                "archived Git tree/blob correspondence; the original report's two "
                "inventory passes used ordinary path opens and recorded the deployed "
                "tree identity from the already sealed creation record."
            ),
        },
    }
    completion["artifact_sha256"] = _artifact_sha256(completion)
    _exclusive_json(OUTPUT, completion)
    written, record = _sealed_json(
        OUTPUT,
        file_sha256=_stable_file(OUTPUT)["sha256"],
        artifact_sha256=completion["artifact_sha256"],
    )
    _require(written == completion, "written completion changed")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
