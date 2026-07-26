#!/usr/bin/env python3
"""Aggregate the frozen 21-case static-scene gauge source transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.phystwin_static_scene_gauge_source import (
    StaticSceneGaugeSourceGate,
    aggregate_phystwin_static_scene_gauge_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cases = protocol["cohort"]["transfer_cases"]
    gate = StaticSceneGaugeSourceGate(**protocol["source_transfer_gate"])
    result = aggregate_phystwin_static_scene_gauge_source(
        [
            args.result_root / case / "prefix_competence.json"
            for case in cases
        ],
        expected_cases=cases,
        gate=gate,
    )
    result["protocol_id"] = protocol["protocol_id"]
    result["protocol_sha256"] = protocol["protocol_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
