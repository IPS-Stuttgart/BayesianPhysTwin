#!/usr/bin/env python3
"""Verify and sanitize one common retained Deform360 production traceback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, cast

_FAILURE_SCHEMA = (
    "bayesian-phystwin.deform360-calibration-visual-technical-failure"
)
_DIAGNOSTIC_SCHEMA = (
    "bayesian-phystwin.deform360-visual-production-failure-diagnostic"
)
_EXPECTED_PREDECESSOR_BOUNDARY = {
    "calibration_robot_state_opened": False,
    "calibration_tactile_payloads_opened": False,
    "confirmation_payloads_opened": False,
    "motioncrafter_prediction_payloads_opened": True,
    "replacement_allowed": False,
    "reserved_evaluation_frames_opened": False,
    "retained_calibration_camera_payloads_opened": True,
    "target_outcomes_used": False,
}
_FILE_PATTERN = re.compile(r'^(\s*File ")([^"]+)(", line \d+, in .+)$')
_FRAME_PATTERN = re.compile(r'File "([^"]+)", line (\d+), in (.+)$')


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _literal_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with literal string keys")
    return cast(dict[str, Any], value)


def _lower_hex(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _nonempty_literal(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be one nonempty literal string")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _reject_symlinks(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ValueError(f"path contains a symbolic link: {candidate}")


def _stable_read(path: Path, *, expected_size: int | None = None) -> bytes:
    _reject_symlinks(path)
    flags = (
        os.O_RDONLY
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        if expected_size is not None and before.st_size != expected_size:
            raise ValueError(f"unexpected byte count for {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"file changed while being read: {path}")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ValueError(f"short read for {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_stable_read(path).decode("utf-8"))
    return _literal_mapping(payload, name=str(path))


def _safe_member(root: Path, relative: object) -> Path:
    value = _nonempty_literal(relative, name="retained relative path")
    if "\\" in value or "\x00" in value:
        raise ValueError("retained relative path must be canonical POSIX")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("retained relative path must be canonical POSIX")
    path = root
    for part in pure.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"retained path contains a symlink: {value}")
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("retained path escapes the production run root")
    return path


def _write_new_text(path: Path, content: str) -> None:
    payload = content.encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError(f"short write for {path}")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sanitize_traceback(
    traceback_text: str,
    *,
    replacements: dict[str, str],
) -> tuple[str, list[dict[str, object]], str]:
    sanitized = traceback_text
    for prefix, token in sorted(
        ((key, value) for key, value in replacements.items() if key),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        sanitized = sanitized.replace(prefix, token)

    original_lines = sanitized.splitlines()
    sanitized_lines: list[str] = []
    frames: list[dict[str, object]] = []
    for index, line in enumerate(original_lines):
        match = _FILE_PATTERN.match(line)
        if match:
            original_path = match.group(2)
            if original_path.startswith("<"):
                safe_path = original_path
            else:
                parts = PurePosixPath(original_path).parts
                safe_path = "<ABSOLUTE_PATH>/" + "/".join(parts[-4:])
            line = match.group(1) + safe_path + match.group(3)
            frame_match = _FRAME_PATTERN.search(line)
            if frame_match:
                source_line = (
                    original_lines[index + 1].strip()
                    if index + 1 < len(original_lines)
                    else ""
                )
                frames.append(
                    {
                        "path": frame_match.group(1),
                        "line": int(frame_match.group(2)),
                        "function": frame_match.group(3),
                        "source_line": source_line,
                    }
                )
        sanitized_lines.append(line)

    sanitized = "\n".join(sanitized_lines).rstrip() + "\n"
    nonempty = [line for line in sanitized_lines if line.strip()]
    if not nonempty or not nonempty[0].startswith("Traceback"):
        raise ValueError("retained stderr is not a Python traceback")
    return sanitized, frames, nonempty[-1]


def diagnose_failure(
    *,
    visual_output_root: Path,
    admission_id: str,
    implementation_revision: str,
    attempt_id: str,
    failed_workflow_run_id: int,
    expected_job_count: int,
    expected_stderr_sha256: str,
    expected_stderr_bytes: int,
    output_dir: Path,
    path_replacements: dict[str, str] | None = None,
) -> dict[str, object]:
    """Verify all retained receipts and publish one sanitized common traceback."""

    admission_id = _lower_hex(admission_id, name="admission_id")
    implementation_revision = _nonempty_literal(
        implementation_revision,
        name="implementation_revision",
    )
    if len(implementation_revision) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_revision
    ):
        raise ValueError("implementation_revision must be one exact commit SHA")
    attempt_id = _nonempty_literal(attempt_id, name="attempt_id")
    failed_workflow_run_id = _positive_integer(
        failed_workflow_run_id,
        name="failed_workflow_run_id",
    )
    expected_job_count = _positive_integer(
        expected_job_count,
        name="expected_job_count",
    )
    expected_stderr_sha256 = _lower_hex(
        expected_stderr_sha256,
        name="expected_stderr_sha256",
    )
    expected_stderr_bytes = _positive_integer(
        expected_stderr_bytes,
        name="expected_stderr_bytes",
    )

    visual_output_root = visual_output_root.absolute()
    run_root = visual_output_root / admission_id / implementation_revision
    failure_root = run_root / "failures"
    _reject_symlinks(run_root)
    if not failure_root.is_dir():
        raise ValueError("retained failure directory is missing")
    receipt_paths = sorted(failure_root.glob("*.json"))
    if len(receipt_paths) != expected_job_count:
        raise ValueError(
            f"expected {expected_job_count} failure receipts, "
            f"found {len(receipt_paths)}"
        )

    descriptors: set[tuple[str, int]] = set()
    detail_hashes: set[str] = set()
    job_ids: set[str] = set()
    representative: dict[str, Any] | None = None
    for receipt_path in receipt_paths:
        receipt = _load_json(receipt_path)
        if receipt.get("schema") != _FAILURE_SCHEMA:
            raise ValueError("retained failure schema changed")
        if receipt.get("admission_id") != admission_id:
            raise ValueError("retained failure admission changed")
        if receipt.get("implementation_revision") != implementation_revision:
            raise ValueError("retained failure implementation changed")
        if receipt.get("attempt_id") != attempt_id:
            raise ValueError("retained failure attempt changed")
        if receipt.get("stage") != "motioncrafter-production":
            raise ValueError("retained failure stage changed")
        if receipt.get("return_code") != 1:
            raise ValueError("retained failure return code changed")
        if receipt.get("completion_kind") != "technical_failure":
            raise ValueError("retained failure completion kind changed")
        if receipt.get("status") != "failed":
            raise ValueError("retained failure status changed")
        boundary = _literal_mapping(
            receipt.get("information_boundary"),
            name="retained failure information boundary",
        )
        if boundary != _EXPECTED_PREDECESSOR_BOUNDARY:
            raise ValueError("retained failure information boundary changed")
        stderr = _literal_mapping(
            receipt.get("stderr"),
            name="retained stderr descriptor",
        )
        descriptor = (
            _lower_hex(stderr.get("sha256"), name="stderr sha256"),
            _positive_integer(stderr.get("byte_count"), name="stderr byte_count"),
        )
        descriptors.add(descriptor)
        detail_hashes.add(
            _lower_hex(receipt.get("detail_sha256"), name="detail_sha256")
        )
        job_id = _lower_hex(receipt.get("job_id"), name="job_id")
        if job_id in job_ids:
            raise ValueError("retained failure job identities contain duplicates")
        job_ids.add(job_id)
        if representative is None:
            representative = receipt

    expected_descriptor = {(expected_stderr_sha256, expected_stderr_bytes)}
    if descriptors != expected_descriptor:
        raise ValueError(
            "retained failures do not share the expected traceback: "
            f"{sorted(descriptors)!r}"
        )
    if len(detail_hashes) != 1:
        raise ValueError("retained failures do not share one detail identity")
    if representative is None:
        raise ValueError("no representative failure exists")

    stderr_record = _literal_mapping(
        representative.get("stderr"),
        name="representative stderr descriptor",
    )
    stderr_relative = _nonempty_literal(
        stderr_record.get("path"),
        name="representative stderr path",
    )
    stderr_parts = PurePosixPath(stderr_relative).parts
    if len(stderr_parts) != 3 or stderr_parts[:2] != ("logs", attempt_id):
        raise ValueError("representative stderr path is outside the exact attempt")
    stderr_path = _safe_member(run_root, stderr_relative)
    stderr_bytes = _stable_read(
        stderr_path,
        expected_size=expected_stderr_bytes,
    )
    observed_sha = _sha256(stderr_bytes)
    if observed_sha != expected_stderr_sha256:
        raise ValueError("representative traceback digest changed")
    traceback_text = stderr_bytes.decode("utf-8", errors="strict")

    replacements = dict(path_replacements or {})
    replacements.setdefault(str(visual_output_root.parent.parent.parent), "<DEFORM360_STORAGE>")
    sanitized, frames, exception_line = _sanitize_traceback(
        traceback_text,
        replacements=replacements,
    )

    diagnostic = {
        "schema": _DIAGNOSTIC_SCHEMA,
        "schema_version": 1,
        "failed_workflow_run_id": failed_workflow_run_id,
        "failed_attempt_id": attempt_id,
        "implementation_revision": implementation_revision,
        "admission_id": admission_id,
        "failure_receipt_count": len(receipt_paths),
        "unique_job_count": len(job_ids),
        "common_stage": "motioncrafter-production",
        "common_return_code": 1,
        "common_stderr_sha256": observed_sha,
        "common_stderr_bytes": len(stderr_bytes),
        "common_detail_sha256": next(iter(detail_hashes)),
        "exception_line": exception_line,
        "frames": frames,
        "sanitized_traceback_sha256": _sha256(sanitized.encode("utf-8")),
        "predecessor_information_boundary": _EXPECTED_PREDECESSOR_BOUNDARY,
        "diagnostic_information_boundary": {
            "retained_technical_logs_opened": True,
            "retained_calibration_camera_payloads_opened_by_diagnostic": False,
            "calibration_tactile_payloads_opened_by_diagnostic": False,
            "calibration_robot_state_opened_by_diagnostic": False,
            "reserved_evaluation_frames_opened_by_diagnostic": False,
            "adaptive_confirmation_payloads_opened_by_diagnostic": False,
            "confirmation_payloads_opened_by_diagnostic": False,
            "target_outcomes_used_by_diagnostic": False,
            "replacement_allowed": False,
        },
    }

    output_dir = output_dir.absolute()
    _reject_symlinks(output_dir.parent)
    output_dir.mkdir(parents=True, exist_ok=False)
    diagnostic_text = json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
    _write_new_text(output_dir / "diagnostic.json", diagnostic_text)
    _write_new_text(output_dir / "sanitized-traceback.txt", sanitized)
    print("=== Sanitized common traceback ===")
    print(sanitized, end="")
    print("=== End sanitized common traceback ===")
    print(diagnostic_text, end="")
    return diagnostic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-output-root", type=Path, required=True)
    parser.add_argument("--admission-id", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--failed-workflow-run-id", type=int, required=True)
    parser.add_argument("--expected-job-count", type=int, required=True)
    parser.add_argument("--expected-stderr-sha256", required=True)
    parser.add_argument("--expected-stderr-bytes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    replacements = {
        os.environ.get("GITHUB_WORKSPACE", ""): "<GITHUB_WORKSPACE>",
        os.environ.get("RUNNER_TEMP", ""): "<RUNNER_TEMP>",
        os.environ.get("HOME", ""): "<HOME>",
        "/mnt/lexar4tb/datasets/deform360": "<DEFORM360_STORAGE>",
    }
    diagnose_failure(
        visual_output_root=args.visual_output_root,
        admission_id=args.admission_id,
        implementation_revision=args.implementation_revision,
        attempt_id=args.attempt_id,
        failed_workflow_run_id=args.failed_workflow_run_id,
        expected_job_count=args.expected_job_count,
        expected_stderr_sha256=args.expected_stderr_sha256,
        expected_stderr_bytes=args.expected_stderr_bytes,
        output_dir=args.output_dir,
        path_replacements=replacements,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
