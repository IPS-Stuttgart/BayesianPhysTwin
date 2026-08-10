"""Materialize the frozen Deform360 source-support failure census input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Final, cast

from bayesian_phystwin.deform360_provider_failure_census_v1 import (
    validate_deform360_provider_failure_census_payload,
)
from bayesian_phystwin.deform360_source_support_failure_evidence_v1 import (
    DEFORM360_SOURCE_SUPPORT_AGGREGATION_POLICY_ID,
    FROZEN_DEFORM360_SOURCE_SUPPORT_EVIDENCE_LOCK_V1,
    Deform360SourceSupportEvidenceLockV1,
    build_deform360_source_support_failure_evidence_v1,
)
from bayesian_phystwin.provider_failure_report_io import (
    canonical_json_sha256,
    load_provider_failure_input,
)

EVIDENCE_FILENAME: Final = "provider-failure-evidence.json"
RECEIPT_FILENAME: Final = "materialization-receipt.json"
CHECKSUM_FILENAME: Final = "SHA256SUMS"
DEFAULT_OUTPUT_RELATIVE_DIRECTORY: Final = (
    "bayesian-phystwin/deform360-provider-failure-evidence-v1"
)
MATERIALIZATION_RECEIPT_SCHEMA: Final = (
    "bayesian_phystwin.deform360_source_support_failure_evidence_materialization"
)
MATERIALIZATION_RECEIPT_VERSION: Final = 1
MAXIMUM_METRIC_BATCH_BYTES: Final = 64 * 1024 * 1024


def _canonical_json_bytes(value: object, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _ordinary_directory(path: str | Path, *, name: str) -> Path:
    candidate = Path(path)
    before = candidate.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{name} must be an ordinary directory")
    resolved = candidate.resolve(strict=True)
    after = resolved.stat()
    if not stat.S_ISDIR(after.st_mode):
        raise ValueError(f"{name} must resolve to a directory")
    return resolved


def _ordinary_file_below(root: Path, relative: str, *, name: str) -> Path:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or path.as_posix() != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a canonical relative path")
    candidate = root.joinpath(*path.parts)
    metadata = candidate.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be an ordinary file")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate or root not in resolved.parents:
        raise ValueError(f"{name} escapes the results root")
    return resolved


def _write_file(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _checksums(directory: Path) -> dict[str, str]:
    return {
        name: _sha256_file(directory / name)
        for name in (EVIDENCE_FILENAME, RECEIPT_FILENAME)
    }


def _checksum_bytes(checksums: dict[str, str]) -> bytes:
    return "".join(
        f"{checksums[name]}  {name}\n"
        for name in (EVIDENCE_FILENAME, RECEIPT_FILENAME)
    ).encode("utf-8")


def _verify_published_directory(
    directory: Path,
    *,
    evidence_bytes: bytes,
    receipt_bytes: bytes,
) -> None:
    root = _ordinary_directory(directory, name="published evidence directory")
    expected_names = {EVIDENCE_FILENAME, RECEIPT_FILENAME, CHECKSUM_FILENAME}
    actual_names = {entry.name for entry in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError("published evidence member roster changed")
    evidence_path = _ordinary_file_below(root, EVIDENCE_FILENAME, name="evidence")
    receipt_path = _ordinary_file_below(root, RECEIPT_FILENAME, name="receipt")
    checksum_path = _ordinary_file_below(root, CHECKSUM_FILENAME, name="checksums")
    if evidence_path.read_bytes() != evidence_bytes:
        raise ValueError("published evidence bytes changed")
    if receipt_path.read_bytes() != receipt_bytes:
        raise ValueError("published receipt bytes changed")
    expected_checksums = _checksums(root)
    if checksum_path.read_bytes() != _checksum_bytes(expected_checksums):
        raise ValueError("published checksums changed")


def _receipt(
    *,
    lock: Deform360SourceSupportEvidenceLockV1,
    evidence_relative_path: str,
    evidence_bytes: bytes,
    report: dict[str, object],
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": MATERIALIZATION_RECEIPT_SCHEMA,
        "schema_version": MATERIALIZATION_RECEIPT_VERSION,
        "builder_policy_id": DEFORM360_SOURCE_SUPPORT_AGGREGATION_POLICY_ID,
        **lock.metadata(),
        "evidence_relative_path": evidence_relative_path,
        "evidence_sha256": _sha256_bytes(evidence_bytes),
        "evidence_bytes": len(evidence_bytes),
        "evidence_content_sha256": cast(str, report["input_content_sha256"]),
        "provider_id": report["provider_id"],
        "statistical_unit": "physical-object",
        "record_count": report["record_count"],
        "accepted_count": report["accepted_count"],
        "classified_rejection_count": report["classified_rejection_count"],
        "unresolved_rejection_count": report["unresolved_rejection_count"],
        "primary_category_counts": report["primary_category_counts"],
        "information_boundary": {
            "source_only": True,
            "raw_tree_traversal_allowed": False,
            "confirmation_payloads_opened": False,
            "adaptive_confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "future_frames_used": False,
            "replacement_allowed": False,
        },
    }
    receipt["receipt_id"] = canonical_json_sha256(receipt)
    return receipt


def materialize_deform360_source_support_failure_evidence_v1(
    results_root: str | Path,
    *,
    lock: Deform360SourceSupportEvidenceLockV1 = (
        FROZEN_DEFORM360_SOURCE_SUPPORT_EVIDENCE_LOCK_V1
    ),
    output_relative_directory: str = DEFAULT_OUTPUT_RELATIVE_DIRECTORY,
) -> dict[str, object]:
    """Publish one content-addressed, retry-safe equal-object evidence directory."""

    root = _ordinary_directory(results_root, name="Deform360 results root")
    metric_batch_path = _ordinary_file_below(
        root,
        lock.metric_batch_relative_path,
        name="frozen metric batch",
    )
    payload, artifact = load_provider_failure_input(
        metric_batch_path,
        maximum_input_bytes=MAXIMUM_METRIC_BATCH_BYTES,
    )
    if artifact["sha256"] != lock.metric_batch_sha256:
        raise ValueError("frozen metric-batch SHA-256 changed")
    if artifact["bytes"] != lock.metric_batch_bytes:
        raise ValueError("frozen metric-batch byte count changed")

    evidence = build_deform360_source_support_failure_evidence_v1(
        payload,
        lock=lock,
    )
    report = validate_deform360_provider_failure_census_payload(evidence)
    evidence_bytes = _canonical_json_bytes(evidence)
    evidence_sha256 = _sha256_bytes(evidence_bytes)

    output_path = PurePosixPath(output_relative_directory)
    if (
        output_path.is_absolute()
        or output_path.as_posix() != output_relative_directory
        or any(part in {"", ".", ".."} for part in output_path.parts)
    ):
        raise ValueError("output_relative_directory must be canonical and relative")
    output_root = root.joinpath(*output_path.parts)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = _ordinary_directory(output_root, name="evidence output root")
    if root not in output_root.parents:
        raise ValueError("evidence output root escapes the results root")

    target = output_root / evidence_sha256
    evidence_relative_path = (
        target / EVIDENCE_FILENAME
    ).relative_to(root).as_posix()
    receipt = _receipt(
        lock=lock,
        evidence_relative_path=evidence_relative_path,
        evidence_bytes=evidence_bytes,
        report=report,
    )
    receipt_bytes = _canonical_json_bytes(receipt, indent=2)

    reused_existing = target.exists()
    if reused_existing:
        _verify_published_directory(
            target,
            evidence_bytes=evidence_bytes,
            receipt_bytes=receipt_bytes,
        )
    else:
        temporary = output_root / f".{evidence_sha256}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            _write_file(temporary / EVIDENCE_FILENAME, evidence_bytes)
            _write_file(temporary / RECEIPT_FILENAME, receipt_bytes)
            checksums = _checksums(temporary)
            _write_file(temporary / CHECKSUM_FILENAME, _checksum_bytes(checksums))
            _sync_directory(temporary)
            try:
                os.rename(temporary, target)
            except FileExistsError:
                _verify_published_directory(
                    target,
                    evidence_bytes=evidence_bytes,
                    receipt_bytes=receipt_bytes,
                )
                reused_existing = True
            _sync_directory(output_root)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    _verify_published_directory(
        target,
        evidence_bytes=evidence_bytes,
        receipt_bytes=receipt_bytes,
    )

    return {
        "schema": MATERIALIZATION_RECEIPT_SCHEMA,
        "schema_version": MATERIALIZATION_RECEIPT_VERSION,
        "materialization_directory_relative": target.relative_to(root).as_posix(),
        "evidence_relative_path": evidence_relative_path,
        "evidence_sha256": evidence_sha256,
        "evidence_bytes": len(evidence_bytes),
        "evidence_content_sha256": report["input_content_sha256"],
        "materialization_receipt_id": receipt["receipt_id"],
        "builder_policy_id": DEFORM360_SOURCE_SUPPORT_AGGREGATION_POLICY_ID,
        "record_count": report["record_count"],
        "classified_rejection_count": report["classified_rejection_count"],
        "unresolved_rejection_count": report["unresolved_rejection_count"],
        "primary_category_counts": report["primary_category_counts"],
        "reused_existing": reused_existing,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Existing /mnt/.../deform360/results directory",
    )
    parser.add_argument(
        "--output-relative-directory",
        default=DEFAULT_OUTPUT_RELATIVE_DIRECTORY,
        help="Canonical destination below the results directory",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional no-overwrite JSON summary path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = materialize_deform360_source_support_failure_evidence_v1(
        args.results_root,
        output_relative_directory=args.output_relative_directory,
    )
    serialized = _canonical_json_bytes(summary, indent=2)
    if args.summary_output is not None:
        summary_output = args.summary_output.resolve(strict=False)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        _write_file(summary_output, serialized)
    print(serialized.decode("utf-8"), end="")
    return 0


__all__ = [
    "CHECKSUM_FILENAME",
    "DEFAULT_OUTPUT_RELATIVE_DIRECTORY",
    "EVIDENCE_FILENAME",
    "MATERIALIZATION_RECEIPT_SCHEMA",
    "MATERIALIZATION_RECEIPT_VERSION",
    "RECEIPT_FILENAME",
    "materialize_deform360_source_support_failure_evidence_v1",
]


if __name__ == "__main__":
    raise SystemExit(main())
