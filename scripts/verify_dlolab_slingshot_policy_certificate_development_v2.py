#!/usr/bin/env python3
"""Verify the posterior-aware Slingshot source-development artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_builder() -> ModuleType:
    path = ROOT / "scripts/audit_dlolab_slingshot_policy_certificate_development_v2.py"
    spec = importlib.util.spec_from_file_location("policy_certificate_v2_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load policy-certificate development builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_id(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "artifact_id"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.summary.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("development summary must contain a JSON object")
    if value.get("artifact_id") != _canonical_id(value):
        raise ValueError("development summary artifact_id changed")
    expected = _load_builder().build()
    if value != expected:
        raise ValueError("development summary does not reproduce")
    if (
        value.get("advancement_gate_passed") is not True
        or value.get("prospective_coverage_claim") is not False
        or value.get("prefix_panel_outcomes_read") is not False
        or value.get("new_simulation_executed") is not False
    ):
        raise ValueError("development claim boundary changed")
    print(
        json.dumps(
            {
                "artifact_id": value["artifact_id"],
                "selected_model": value["selected_model"],
                "advancement_gate_passed": value["advancement_gate_passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
