#!/usr/bin/env python3
"""Run the frozen target-free simulation-based calibration study."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from bayesian_phystwin.strict_json_report_io import load_strict_json_mapping
from bayesian_phystwin_experiments.simulation_based_calibration_v1 import (
    compact_summary,
    run_simulation_based_calibration,
)

REGISTERED_DECISION = (
    "exact-model-calibration-not-rejected-and-misspecification-detected"
)


def _load(path: Path) -> dict[str, Any]:
    value, _ = load_strict_json_mapping(
        path,
        artifact_label="simulation-based calibration protocol",
    )
    return dict(value)


def _sync_parent_directory(path: Path) -> None:
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


def _atomic_create(path: Path, payload: dict[str, Any]) -> None:
    requested_path = path.absolute()
    requested_path.parent.mkdir(parents=True, exist_ok=True)
    if requested_path.is_symlink():
        raise FileExistsError(f"refusing to publish through symlink {requested_path}")
    output_path = requested_path.parent.resolve(strict=True) / requested_path.name
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite {output_path}") from error
        _sync_parent_directory(output_path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--require-registered-decision", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_simulation_based_calibration(_load(args.protocol))
    summary = compact_summary(result)
    _atomic_create(args.output, result)
    _atomic_create(args.summary_output, summary)
    print(
        json.dumps(
            {
                "protocol_id": summary["protocol_id"],
                "result_id": summary["result_id"],
                "summary_id": summary["summary_id"],
                "decision": summary["decision"],
                "replicate_row_count": summary["replicate_row_count"],
                "correlated_failed_test_fraction": summary[
                    "correlated_failed_test_fraction"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_registered_decision and summary["decision"] != REGISTERED_DECISION:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
