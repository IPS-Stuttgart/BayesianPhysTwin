#!/usr/bin/env python3
"""Build and qualify the outcome-open Deform360 pairwise source panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_pairwise_regret_guard_source import (
    build_pairwise_regret_source_payload,
    evaluate_pairwise_regret_guard_source,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--measurement-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = build_pairwise_regret_source_payload(
        args.panel_root, args.measurement_root
    )
    qualification = evaluate_pairwise_regret_guard_source(source)
    _write_json(args.output_dir / "source_payload.json", source)
    _write_json(args.output_dir / "source_qualification.json", qualification)
    print(
        json.dumps(
            {
                "source_gate_passed": qualification["source_gate_passed"],
                "fresh_accuracy_evaluation_allowed": qualification[
                    "fresh_accuracy_evaluation_allowed"
                ],
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
