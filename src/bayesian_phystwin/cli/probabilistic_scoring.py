"""Score registered predictive distributions with proper scoring rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.decisive_evidence import parse_decisive_evidence
from bayesian_phystwin.probabilistic_scoring import (
    build_decisive_evidence_from_score_report,
    score_probabilistic_bundle,
)

DEFAULT_MAXIMUM_INPUT_BYTES: Final = 64 * 1024 * 1024


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"input JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"input JSON contains non-finite constant {value!r}")


def _signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _load_input(
    path: Path,
    *,
    maximum_bytes: int,
) -> tuple[Mapping[str, object], dict[str, object]]:
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int):
        raise TypeError("maximum_input_bytes must be a genuine integer")
    if maximum_bytes < 1:
        raise ValueError("maximum_input_bytes must be positive")
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        try:
            candidate_metadata = os.stat(candidate, follow_symlinks=False)
        except OSError as error:
            raise ValueError("probabilistic-score input path is unreadable") from error
        if stat.S_ISLNK(candidate_metadata.st_mode):
            raise ValueError("probabilistic-score input path must not contain symlinks")
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise ValueError("probabilistic-score input is unreadable") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("probabilistic-score input must be an ordinary file")
    if before.st_size > maximum_bytes:
        raise ValueError("probabilistic-score input exceeds its byte budget")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError("probabilistic-score input is unreadable") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("probabilistic-score input must be an ordinary file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("probabilistic-score input changed while it was opened")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise ValueError("probabilistic-score input exceeds its byte budget")
    try:
        after_path = os.lstat(absolute)
    except OSError as error:
        raise ValueError(
            "probabilistic-score input changed while it was read"
        ) from error
    if (
        _signature(before) != _signature(after_descriptor)
        or _signature(before) != _signature(after_path)
        or len(payload) != before.st_size
    ):
        raise ValueError("probabilistic-score input changed while it was read")
    source = absolute.resolve(strict=True)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("probabilistic-score input must be UTF-8 JSON") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("probabilistic-score input is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("probabilistic-score input root must be a JSON object")
    return cast(Mapping[str, object], value), {
        "path": str(source),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("report_json", type=Path)
    parser.add_argument(
        "--evidence-json",
        type=Path,
        help=(
            "optional decisive-evidence bundle for subsequent "
            "`bpt evidence summarize` analysis"
        ),
    )
    parser.add_argument(
        "--maximum-input-bytes",
        type=int,
        default=DEFAULT_MAXIMUM_INPUT_BYTES,
        help="fail when the strict input exceeds this byte budget",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace existing outputs instead of failing closed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload, input_artifact = _load_input(
        args.input_json,
        maximum_bytes=args.maximum_input_bytes,
    )
    score_report = score_probabilistic_bundle(payload)
    published_report = {
        **score_report,
        "input_artifact": input_artifact,
    }
    published_report["status_sha256"] = content_id(published_report)
    write_atomic_json(
        published_report,
        args.report_json,
        overwrite=args.overwrite,
    )

    evidence_path: str | None = None
    if args.evidence_json is not None:
        evidence = build_decisive_evidence_from_score_report(score_report)
        parse_decisive_evidence(evidence)
        evidence["source_score_report_id"] = score_report["report_id"]
        evidence["source_score_input_sha256"] = input_artifact["sha256"]
        evidence["status_sha256"] = content_id(evidence)
        write_atomic_json(
            evidence,
            args.evidence_json,
            overwrite=args.overwrite,
        )
        evidence_path = str(args.evidence_json.resolve(strict=False))

    aggregate = score_report["aggregate"]
    if not isinstance(aggregate, Mapping):
        raise AssertionError("score report aggregate changed type")
    score_configuration = score_report["score_configuration"]
    if not isinstance(score_configuration, Mapping):
        raise AssertionError("score report configuration changed type")
    score_names = score_configuration["score_names"]
    if isinstance(score_names, (str, bytes)) or not isinstance(score_names, Sequence):
        raise AssertionError("score report names changed type")
    methods = score_report["methods"]
    rows = score_report["unit_score_rows"]
    if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
        raise AssertionError("score report methods changed type")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise AssertionError("score report rows changed type")
    print(
        json.dumps(
            {
                "status": "written",
                "report": str(args.report_json.resolve(strict=False)),
                "evidence": evidence_path,
                "report_id": score_report["report_id"],
                "protocol_id": score_report["protocol_id"],
                "score_names": list(score_names),
                "method_count": len(methods),
                "unit_score_row_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
