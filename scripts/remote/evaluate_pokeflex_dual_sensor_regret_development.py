#!/usr/bin/env python3
"""Evaluate target-free dual-sensor consensus on opened PokeFlex takes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from bayesian_phystwin.pokeflex_dual_sensor_regret_guard import (  # noqa: E402
    evaluate_pokeflex_dual_sensor_consensus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-improvement-mm", type=float, default=0.0)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.artifacts]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for payload in payloads:
        metadata = payload.get("dual_sensor_regret_development")
        if not isinstance(metadata, dict):
            raise ValueError("dual-sensor development provenance is missing")
        if metadata.get("target_objects_opened") is not False:
            raise ValueError("target-object boundary changed")
    result = evaluate_pokeflex_dual_sensor_consensus(
        payloads,
        minimum_improvement_mm=args.minimum_improvement_mm,
    )
    result["candidate_artifacts"] = [str(path) for path in paths]
    result["target_objects_opened"] = False
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output.resolve()
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"existing development evaluation differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "relative_improvement": result[
                    "object_balanced_relative_improvement"
                ],
                "object_wins": result["object_wins"],
                "object_losses": result["object_losses"],
                "accepted_frame_count": result["accepted_frame_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
