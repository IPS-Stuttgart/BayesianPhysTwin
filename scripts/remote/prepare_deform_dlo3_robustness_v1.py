#!/usr/bin/env python3
"""Bind the prospective DLO3 source panel without opening trajectory payloads."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import run_deform_dlo_source as source_runtime

from bayesian_phystwin_experiments.deform_dlo_robustness import (
    build_deform_dlo3_source_manifest,
    load_deform_dlo_robustness_v1_protocol,
    validate_deform_dlo3_source_manifest,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file
from bayesian_phystwin_experiments.deform_dlo_upstream import (
    load_deform_dlo_initialization,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"source output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    upstream_protocol = _mapping(protocol.get("upstream"), label="upstream")
    data = _mapping(protocol.get("data"), label="data")
    upstream = source_runtime._assert_upstream(
        args.upstream_root, str(upstream_protocol["commit"])
    )
    initialization = load_deform_dlo_initialization(
        args.upstream_root.resolve() / "train_DEFORM.py", "DLO3"
    )
    if (
        initialization.node_count != int(cast(Any, data["node_count"]))
        or initialization.coordinate_transform != data["coordinate_transform"]
        or initialization.source_sha256 != str(upstream_protocol["train_script_sha256"])
    ):
        raise RuntimeError("DLO3 initialization differs from the locked protocol")

    data_root = args.upstream_root.resolve() / "data_set"
    manifest = build_deform_dlo3_source_manifest(protocol_path, data_root)
    manifest_path = output_root / "source_manifest.json"
    _write_json(manifest_path, manifest)
    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(protocol_path),
        verify_files=True,
    )
    receipt = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-source-preflight-v1",
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "upstream": upstream,
        "model_initialization": initialization.to_record(),
        "partition_counts": {name: len(values) for name, values in partitions.items()},
        "trajectory_deserialized": False,
        "source_test_opened": False,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "reserve_payload_enumerated": False,
        "reserve_payload_read": False,
        "prob4d_used": False,
        "held_v8_access": False,
        "next_stage": "source-development-training-only",
    }
    receipt_path = output_root / "preflight.json"
    _write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
