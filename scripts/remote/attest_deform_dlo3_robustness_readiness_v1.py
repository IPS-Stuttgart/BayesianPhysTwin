#!/usr/bin/env python3
"""Bind the all-train method, dry run, environment, and source archive."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path

from bayesian_phystwin_experiments.deform_dlo_robustness import (
    load_deform_dlo_robustness_v1_protocol,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--alltrain-result", type=Path, required=True)
    parser.add_argument("--dry-run-result", type=Path, required=True)
    parser.add_argument("--custody-deviation", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    alltrain_path = args.alltrain_result.resolve()
    dry_run_path = args.dry_run_result.resolve()
    deviation_path = args.custody_deviation.resolve()
    archive_path = args.source_archive.resolve()
    load_deform_dlo_robustness_v1_protocol(protocol_path)
    alltrain = _read_json(alltrain_path)
    dry_run = _read_json(dry_run_path)
    deviation = _read_json(deviation_path)
    revision = str(args.source_revision)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("source revision must be a full lowercase Git commit")
    if (
        alltrain.get("contract") != "deform-dlo3-robustness-alltrain-result-v1"
        or alltrain.get("primary_eval_read") is not False
        or alltrain.get("target_authorized") is not False
        or dry_run.get("contract") != "deform-dlo3-robustness-evaluator-dry-run-v1"
        or dry_run.get("pipeline_passed") is not True
        or dry_run.get("primary_eval_read") is not False
        or dry_run.get("target_authorized") is not False
        or deviation.get("contract") != "deform-dlo3-count-only-custody-deviation-v1"
        or deviation.get("official_eval_read") is not False
        or not archive_path.is_file()
    ):
        raise ValueError("DLO3 readiness inputs differ")
    alltrain_runtime = _mapping(alltrain.get("runtime"), label="alltrain runtime")
    dry_runtime = _mapping(dry_run.get("runtime"), label="dry-run runtime")
    if alltrain_runtime.get("torch") != dry_runtime.get(
        "torch"
    ) or alltrain_runtime.get("cuda") != dry_runtime.get("cuda"):
        raise ValueError("DLO3 readiness runtime differs")
    payload = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-readiness-v1",
        "protocol": _identity(protocol_path),
        "alltrain_result": _identity(alltrain_path),
        "dry_run_result": _identity(dry_run_path),
        "custody_deviation": _identity(deviation_path),
        "source_archive": _identity(archive_path),
        "source_revision": revision,
        "runtime": dict(alltrain_runtime),
        "dry_run_pipeline_passed": True,
        "count_only_custody_deviation_acknowledged": True,
        "target_authorized": True,
        "official_eval_read": False,
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "case_replacement": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked readiness output differs: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
