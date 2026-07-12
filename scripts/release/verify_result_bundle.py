#!/usr/bin/env python3
"""Verify a deterministic result directory against its emitted manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compare_values(
    expected: Any,
    actual: Any,
    *,
    path: str,
    numeric_atol: float,
) -> str | None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return None if expected is actual else f"{path}: {expected!r} != {actual!r}"
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=numeric_atol):
            return None
        return f"{path}: {expected!r} != {actual!r}"
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return f"{path}: object keys differ"
        for key in expected:
            difference = _compare_values(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                numeric_atol=numeric_atol,
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: list lengths differ"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = _compare_values(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                numeric_atol=numeric_atol,
            )
            if difference is not None:
                return difference
        return None
    return None if expected == actual else f"{path}: {expected!r} != {actual!r}"


def _parse_csv_cell(value: str) -> str | float:
    try:
        return float(value)
    except ValueError:
        return value


def _semantic_difference(
    expected_path: Path,
    actual_path: Path,
    *,
    numeric_atol: float,
) -> str | None:
    if expected_path.suffix == ".json":
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
    elif expected_path.suffix == ".csv":
        with expected_path.open(newline="", encoding="utf-8") as handle:
            expected = [
                [_parse_csv_cell(cell) for cell in row] for row in csv.reader(handle)
            ]
        with actual_path.open(newline="", encoding="utf-8") as handle:
            actual = [
                [_parse_csv_cell(cell) for cell in row] for row in csv.reader(handle)
            ]
    else:
        return "binary checksum differs"
    return _compare_values(
        expected,
        actual,
        path=expected_path.name,
        numeric_atol=numeric_atol,
    )


def verify_bundle(
    expected_manifest: Path,
    result_dir: Path,
    *,
    numeric_atol: float = 0.0,
) -> dict[str, object]:
    manifest = json.loads(expected_manifest.read_text(encoding="utf-8"))
    failures = []
    tolerance_matches = []
    for name, expected in manifest["artifacts"].items():
        path = result_dir / name
        if not path.is_file():
            failures.append({"artifact": name, "reason": "missing"})
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != expected["bytes"] or actual_hash != expected["sha256"]:
            expected_path = expected_manifest.parent / name
            difference = None
            if numeric_atol > 0.0 and expected_path.is_file():
                difference = _semantic_difference(
                    expected_path,
                    path,
                    numeric_atol=numeric_atol,
                )
            if difference is None and numeric_atol > 0.0 and expected_path.is_file():
                tolerance_matches.append(name)
                continue
            failures.append(
                {
                    "artifact": name,
                    "reason": "mismatch",
                    "semantic_difference": difference,
                    "expected_bytes": expected["bytes"],
                    "actual_bytes": actual_size,
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": actual_hash,
                }
            )
    return {
        "benchmark": manifest["benchmark"],
        "checked": len(manifest["artifacts"]),
        "numeric_atol": numeric_atol,
        "tolerance_matches": tolerance_matches,
        "passed": not failures,
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_manifest")
    parser.add_argument("result_dir")
    parser.add_argument(
        "--numeric-atol",
        type=float,
        default=0.0,
        help="accept JSON/CSV numeric drift up to this absolute tolerance",
    )
    args = parser.parse_args(argv)
    if args.numeric_atol < 0.0:
        parser.error("--numeric-atol must be non-negative")
    result = verify_bundle(
        Path(args.expected_manifest),
        Path(args.result_dir),
        numeric_atol=args.numeric_atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
