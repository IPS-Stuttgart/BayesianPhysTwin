"""Classify source-only provider and guarded-update failure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from bayesian_phystwin.provider_failure_decomposition import (
    analyze_provider_failure_evidence,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(
    path: Path, payload: Mapping[str, object], *, overwrite: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output instead of failing closed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input_json.resolve()
    output_path = args.output_json.resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("input JSON must contain an object")
    report = analyze_provider_failure_evidence(payload)
    report["input_artifact"] = {
        "path": str(input_path),
        "sha256": _sha256(input_path),
    }
    _atomic_write_json(output_path, report, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(output_path),
                "report_id": report["report_id"],
                "record_count": report["record_count"],
                "accepted_count": report["accepted_count"],
                "classified_rejection_count": report[
                    "classified_rejection_count"
                ],
                "unresolved_rejection_count": report[
                    "unresolved_rejection_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
