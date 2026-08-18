#!/usr/bin/env python3
"""Freeze the three-seed DLO3 source decision without opening evaluation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin_experiments.deform_dlo_robustness import (
    evaluate_deform_dlo3_stability_gate,
    load_deform_dlo_robustness_v1_protocol,
    verify_deform_dlo3_seed_bayesian_artifacts_v1,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    result_paths = tuple(path.resolve() for path in args.seed_result)
    results = [_read_json(path) for path in result_paths]
    gate = evaluate_deform_dlo3_stability_gate(results, protocol)
    bayesian_artifacts = [
        verify_deform_dlo3_seed_bayesian_artifacts_v1(result) for result in results
    ]
    payload = {
        **gate,
        "bayesian_artifacts_verified": True,
        "bayesian_artifact_verifications": bayesian_artifacts,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "seed_results": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in result_paths
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked stability output differs: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
